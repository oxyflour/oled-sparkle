#!/usr/bin/env python3
"""Deterministic OLED-on / AG sparkle benchmark with interchangeable backends.

  torch        : native torch FFT angular-spectrum wave optics
  torchoptics  : TorchOptics Field ASM point-source kernel + torch FFT imager
  chromatix   : Chromatix ASM point-source kernel + JAX FFT imager
  mitsuba      : Mitsuba 3 mesh ray intersections + fixed-aperture Snell quadrature

The first three are the SAME scalar thin-phase-screen wave model. Mitsuba is a
DIFFERENT, ray-optics approximation (no diffraction); agreement of those 3
wave backends is a numerical check, NOT validation of the ray/wave agreement.

All backends are deterministic and apply the same postprocessing to output
images. No standard-compliant camera or IEC 62977-3-9 calibration is claimed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reporting import capture_run, execution_context, model_metadata, utc_now, write_html_report


WAVE_BACKENDS = ('torch', 'torchoptics', 'chromatix')
ALL_BACKENDS = WAVE_BACKENDS + ('mitsuba',)


@dataclass
class Model:
    h_m: np.ndarray
    emission: np.ndarray
    dx_m: float
    em_dx_m: float
    wavelength_m: float
    gap_m: float
    n_stack: float
    n_out: float
    na: float
    source_samples: int
    margin_m: float
    offset_x_m: float
    offset_y_m: float
    threshold: float
    pad: int
    batch: int
    display_pitch_m: float
    roi_margin_m: float
    device: str
    ray_radial: int
    ray_angular: int
    ray_chunk: int
    mesh_stride: int
    mitsuba_variant: str
    progress_every: int

    @property
    def shape(self):
        return self.h_m.shape

    @property
    def padded_shape(self):
        return (self.shape[0] + 2*self.pad, self.shape[1] + 2*self.pad)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--height', default='heightfield.png', help='8-bit or 16-bit grayscale PNG')
    p.add_argument('--height-pitch-um', type=float, required=True, help='height image pixel pitch [um]')
    p.add_argument('--height-range-um', type=float, required=True, help='physical height, white minus black [um]')
    p.add_argument('--emission', default='emission.png', help='linear intensity image (RGB PNG accepted)')
    p.add_argument('--emission-pitch-um', type=float, required=True, help='emission image pixel pitch [um], NOT display pixel pitch')
    p.add_argument('--analysis-pitch-um', type=float, default=50., help='square collected-brightness pixel size for sparkle statistics [um]; area-average the optical grid')
    p.add_argument('--emission-channel', choices=('luma','red','green','blue'), default='luma')
    p.add_argument('--gamma-decode', action='store_true', help='convert sRGB colors to linear intensity first')
    p.add_argument('--emission-offset-x-um', type=float, default=0.)
    p.add_argument('--emission-offset-y-um', type=float, default=0.)
    p.add_argument('--gap-um', type=float, default=500., help='OLED plane to mean AG interface [um]')
    p.add_argument('--wavelength-nm', type=float, default=530., help='vacuum wavelength, single wavelength for now [nm]')
    p.add_argument('--n-stack', type=float, default=1.5, help='single effective index inside stack')
    p.add_argument('--n-out', type=float, default=1., help='index between AG and camera')
    p.add_argument('--na', type=float, default=0.08, help='ideal camera object-side NA')
    p.add_argument('--source-samples', type=int, default=1, help='incoherent fixed source nodes per image texel per axis')
    p.add_argument('--emission-threshold', type=float, default=0.001)
    p.add_argument('--source-margin-um', type=float, default=0.)
    p.add_argument('--max-sources', type=int, default=20000, help='hard cap for wave sources; never subsample silently')
    p.add_argument('--pad-px', type=int, default=64, help='wave FFT zero-padding per edge')
    p.add_argument('--batch', type=int, default=2, help='wave propagations per batch')
    p.add_argument('--backend', choices=ALL_BACKENDS+('all',), default='torch')
    p.add_argument('--backend-workers',type=int,choices=(1,2,3,4),default=1,
                   help='independent backend computations to run concurrently')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto', help='torch FFT device; JAX device via JAX_PLATFORMS')
    p.add_argument('--ray-radial', type=int, default=3, help='Mitsuba fixed pupil radial nodes')
    p.add_argument('--ray-angular', type=int, default=8, help='Mitsuba fixed pupil azimuth nodes')
    p.add_argument('--ray-chunk', type=int, default=65536, help='Mitsuba ray queries per chunk')
    p.add_argument('--mesh-stride', type=int, default=1, help='Mitsuba mesh decimation: 1 keeps every height sample')
    p.add_argument('--mitsuba-variant', default='auto', help='e.g. llvm_ad_rgb or cuda_ad_rgb')
    p.add_argument('--roi-margin-um', type=float, default=100., help='geometric crop from brightness image edges [um]')
    p.add_argument('--progress-every', type=int, default=20, help='wave batch log interval; 0 disables')
    p.add_argument('--out', default='sparkle_results')
    p.add_argument('--skip-unavailable', action='store_true', help='in --backend all mode skip backends missing dependencies')
    p.add_argument('--dry-run', action='store_true', help='report image sizes and source estimates without computing')
    return p.parse_args(argv)


def read_height(path, height_range_um):
    im = Image.open(path)
    raw = np.asarray(im)
    if raw.ndim != 2:
        raise ValueError('heightfield must be a SINGLE-CHANNEL grayscale PNG, not RGB')
    if raw.dtype == np.uint8:
        h = raw.astype(np.float32) / 255.
    elif raw.dtype == np.uint16 or im.mode.startswith('I;16') or im.mode == 'I':
        h = raw.astype(np.float32) / 65535.
    else:
        raise ValueError('heightfield PNG must be 8-bit or 16-bit grayscale')
    if not np.isfinite(h).all():
        raise ValueError('heightfield contains invalid pixels')
    return np.clip(h, 0, 1) * height_range_um * 1e-6


def read_emission(path, channel='luma', gamma_decode=False):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    if gamma_decode:
        rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb+.055)/1.055)**2.4)
    if channel == 'red': return rgb[...,0].astype(np.float32)
    if channel == 'green': return rgb[...,1].astype(np.float32)
    if channel == 'blue': return rgb[...,2].astype(np.float32)
    return (rgb[...,0]*.2126 + rgb[...,1]*.7152 + rgb[...,2]*.0722).astype(np.float32)


def model_from_args(a):
    positive = ('height-pitch-um','height-range-um','emission-pitch-um','analysis-pitch-um',
                'gap-um','wavelength-nm','n-stack','n-out','na')
    for k in positive:
        v=getattr(a,k.replace('-','_'))
        if not math.isfinite(v) or v<=0: raise ValueError(f'--{k} must be positive and finite')
    if a.na >= min(a.n_stack,a.n_out):
        raise ValueError('NA must be less than both refractive indices')
    for k in ('source_samples','batch','ray_radial','ray_angular','ray_chunk','mesh_stride'):
        if getattr(a,k)<1: raise ValueError(f'{k} must be >=1')
    if a.roi_margin_um < 0 or a.pad_px < 0:
        raise ValueError('negative ROI margin/padding is not allowed')
    h = read_height(a.height,a.height_range_um)
    em = read_emission(a.emission,a.emission_channel,a.gamma_decode)
    if h.shape[0]<8 or h.shape[1]<8: raise ValueError('heightfield too small (<8 pixels)')
    if max(em.shape)>100_000: raise ValueError('emission image too large')
    device=a.device
    if device=='auto':
        try:
            import torch
            device='cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError: device='cpu'
    return Model(h,em,a.height_pitch_um*1e-6,a.emission_pitch_um*1e-6,
                 a.wavelength_nm*1e-9,a.gap_um*1e-6,a.n_stack,a.n_out,a.na,
                 a.source_samples,a.source_margin_um*1e-6,
                 a.emission_offset_x_um*1e-6,a.emission_offset_y_um*1e-6,
                 a.emission_threshold,a.pad_px,a.batch,a.analysis_pitch_um*1e-6,
                 a.roi_margin_um*1e-6,device,a.ray_radial,a.ray_angular,
                 a.ray_chunk,a.mesh_stride,a.mitsuba_variant,a.progress_every)


def enumerate_sources(m):
    row,col=np.nonzero(m.emission>m.threshold)
    ey,ex=m.emission.shape
    hy,hx=m.shape
    sources=[]
    for sy in range(m.source_samples):
        for sx in range(m.source_samples):
            yy=(row+(sy+.5)/m.source_samples-ey/2)*m.em_dx_m+m.offset_y_m
            xx=(col+(sx+.5)/m.source_samples-ex/2)*m.em_dx_m+m.offset_x_m
            take=(np.abs(yy)<hy*m.dx_m/2 + m.margin_m) & (np.abs(xx)<hx*m.dx_m/2 + m.margin_m)
            # Nearest height sample is allowed; sub-sample positions smaller than
            # dx are quantized. Report this approximation clearly.
            rr=np.rint(yy[take]/m.dx_m+(hy-1)/2).astype(np.int32)
            cc=np.rint(xx[take]/m.dx_m+(hx-1)/2).astype(np.int32)
            weights=m.emission[row[take],col[take]]/m.source_samples**2
            sources.extend(zip(rr.tolist(), cc.tolist(), weights.tolist()))
    if not sources:
        raise ValueError('No positive emission texels overlap the height field')
    # Avoid silently wrapping emitters outside the padded FFT grid.
    for rr, cc, _ in sources:
        if not (0 <= rr+m.pad < hy+2*m.pad and 0 <= cc+m.pad < hx+2*m.pad):
            raise ValueError('Emission source outside padded FFT grid; '
                             'increase --pad-px or reduce --source-margin-um')
    return sources


def py_torch_frequencies(ny,nx,dx,wl,n,gap,na,device):
    import torch
    fy=torch.fft.fftfreq(ny,d=dx,device=device)
    fx=torch.fft.fftfreq(nx,d=dx,device=device)
    q=fy[:,None].square()+fx[None,:].square()
    kz2=(n/wl)**2-q
    kz=torch.sqrt(kz2.clamp(min=0))
    # Remove global carrier to reduce float32 phase cancellation.
    fwd=torch.polar(torch.ones_like(kz), 2*math.pi*gap*(kz-n/wl))
    propagating=(kz2>=0)
    fwd=(fwd*propagating).to(torch.complex64)
    pupil=(q<=(na/wl)**2)
    back=torch.conj(fwd)*pupil
    return fwd,back,pupil


def torch_point_kernel(m,backendsel,ny,nx,fwd,device):
    import torch
    if backendsel=='torch':
        return torch.roll(torch.fft.ifft2(fwd),shifts=(ny//2,nx//2),dims=(-2,-1))
    try:
        from torchoptics import Field, set_default_dtype
    except ImportError as exc:
        raise RuntimeError('TorchOptics is not installed: uv pip install torchoptics') from exc
    set_default_dtype(torch.float32)
    source=torch.zeros((ny,nx),dtype=torch.complex64,device=device)
    source[ny//2,nx//2]=1.+0j
    # In a homogeneous index n, effective wavelength is lambda_vacuum / n.
    field=Field(source,z=0.,wavelength=m.wavelength_m/m.n_stack,
                spacing=(m.dx_m,m.dx_m)).to(device)
    return field.propagate_to_z(m.gap_m,propagation_method='ASM',asm_pad=(0,0)).data.to(torch.complex64)


def simulate_torch_wave(m, backend, max_sources):
    import torch
    src=enumerate_sources(m)
    if len(src)>max_sources:
        raise ValueError(f'{len(src)} source modes exceed --max-sources={max_sources}; '
                         'increase cap only if you accept the FFT cost')
    h,w=m.shape
    H,W=m.padded_shape
    p=m.pad
    fwd,back,pupil=py_torch_frequencies(H,W,m.dx_m,m.wavelength_m,m.n_stack,m.gap_m,m.na,m.device)
    kernel=torch_point_kernel(m,backend,H,W,fwd,m.device)
    hc=torch.as_tensor(m.h_m-np.mean(m.h_m),device=m.device,dtype=torch.float32)
    phase=(2*math.pi/m.wavelength_m)*(m.n_stack-m.n_out)*hc
    mod=torch.ones((H,W),device=m.device,dtype=torch.complex64)
    mod[p:p+h,p:p+w]=torch.polar(torch.ones_like(phase),phase)
    img=torch.zeros((h,w),dtype=torch.float32,device=m.device)
    for start in range(0,len(src),m.batch):
        batch=src[start:start+m.batch]
        shifted=[]
        for y,x,wt in batch:
            yy,xx=y+p,x+p
            shifted.append(torch.roll(kernel,shifts=(yy-H//2,xx-W//2),dims=(-2,-1)))
        fields=torch.stack(shifted)*mod
        camera=torch.fft.ifft2(torch.fft.fft2(fields,dim=(-2,-1))*back,dim=(-2,-1))
        weights=torch.tensor([s[2] for s in batch],dtype=torch.float32,device=m.device)[:,None,None]
        img+=(camera[:,p:p+h,p:p+w].abs().square()*weights).sum(dim=0)
        if m.progress_every and ((start//m.batch+1)%m.progress_every==0 or start+m.batch>=len(src)):
            print(f'  {backend}: {min(start+m.batch,len(src))}/{len(src)} modes',flush=True)
    return img.cpu().numpy(),len(src)


def simulate_chromatix(m,max_sources):
    """Chromatix ASM point kernel, followed by native JAX FFT imaging.

    Uses micron units internally for numerical stability of Chromatix's
    complex64 wave propagation (e.g. spectrum=0.53, dx=0.5, gap=500).
    """
    try:
        import jax
        import jax.numpy as jnp
        import chromatix.functional as cx
    except ImportError as exc:
        raise RuntimeError('Chromatix/JAX missing: install JAX for your platform and '
                           'uv pip install "git+https://github.com/chromatix-team/chromatix.git'
                           '@ce1482906fc663298613bb252bbf425e3be59839" '
                           '(Python 3.12+; PyPI chromatix 0.0.1 lacks functional APIs)') from exc
    src=enumerate_sources(m)
    if len(src)>max_sources:
        raise ValueError(f'{len(src)} source modes exceed --max-sources={max_sources}')
    h,w=m.shape
    H,W=m.padded_shape
    p=m.pad
    pitch_um=m.dx_m*1e6
    wl_um=m.wavelength_m*1e6
    gap_um=m.gap_m*1e6
    field=cx.plane_wave(shape=(H,W),dx=pitch_um,spectrum=wl_um)
    # Chromatix 0.6 monochromatic scalar fields have shape (H, W).
    impulse=jnp.zeros_like(field.u).at[H//2,W//2].set(1.+0j)
    field=field.replace(u=impulse)
    propagated=cx.asm_propagate(field,z=gap_um,n=m.n_stack,pad_width=0,remove_evanescent=True)
    kernel=propagated.u.reshape((H,W))
    # The rest is native JAX FFT (instead of sending modes through PyTorch).
    fy=jnp.fft.fftfreq(H,d=pitch_um)
    fx=jnp.fft.fftfreq(W,d=pitch_um)
    q=fy[:,None]**2+fx[None,:]**2
    kz2=(m.n_stack/wl_um)**2-q
    kz=jnp.sqrt(jnp.maximum(kz2,0))
    fwd=jnp.exp(2j*jnp.pi*gap_um*(kz-m.n_stack/wl_um))*(kz2>=0)
    pupil=(q<=(m.na/wl_um)**2)
    back=jnp.conj(fwd)*pupil
    phase=(2*jnp.pi/wl_um)*(m.n_stack-m.n_out)*(jnp.asarray(m.h_m*1e6)-float(np.mean(m.h_m)*1e6))
    mod=jnp.ones((H,W),dtype=jnp.complex64).at[p:p+h,p:p+w].set(jnp.exp(1j*phase))

    @jax.jit
    def image_batch(positions, weights):
        # Keep source shifts as dynamic JAX arrays inside one compiled batch.
        # Rolling with Python integer shifts outside jit compiles a separate
        # kernel for each source position and scales poorly for dense sources.
        center=jnp.asarray((H//2,W//2),dtype=jnp.int32)
        waves=jax.vmap(lambda position: jnp.roll(kernel,position-center,axis=(0,1)))(positions)
        cam=jnp.fft.ifft2(jnp.fft.fft2(waves*mod,axes=(-2,-1))*back,axes=(-2,-1))
        return jnp.sum(jnp.abs(cam[:,p:p+h,p:p+w])**2*weights[:,None,None],axis=0)

    ag=np.zeros((h,w),np.float64)
    for start in range(0,len(src),m.batch):
        batch=src[start:start+m.batch]
        positions=jnp.asarray([(y+p,x+p) for y,x,_ in batch],dtype=jnp.int32)
        weights=jnp.asarray([wt for _,_,wt in batch],dtype=jnp.float32)
        result=image_batch(positions,weights)
        ag+=np.asarray(result,dtype=np.float32)
        if m.progress_every and ((start//m.batch+1)%m.progress_every==0 or start+m.batch>=len(src)):
            print(f'  chromatix: {min(start+m.batch,len(src))}/{len(src)} modes',flush=True)
    return ag.astype(np.float32),len(src)


def create_mitsuba_mesh(mi,m):
    """Mesh from measured HEIGHT samples, with normals directed up (+z).

    Mitsuba coordinates are IN MICROMETERS here, avoiding 32-bit cancellation.
    The OLED emission plane is z=0, AG mean surface z=gap_um.
    """
    h_um=(m.h_m-float(np.mean(m.h_m)))*1e6
    ny,nx=h_um.shape
    ys=np.unique(np.r_[np.arange(0,ny,m.mesh_stride),ny-1])
    xs=np.unique(np.r_[np.arange(0,nx,m.mesh_stride),nx-1])
    local_h=h_um[np.ix_(ys,xs)]
    yy,xx=np.meshgrid(ys,xs,indexing='ij')
    dx=m.dx_m*1e6
    x=((xx-(nx-1)/2)*dx).astype(np.float32)
    y=((yy-(ny-1)/2)*dx).astype(np.float32)
    z=(local_h+m.gap_m*1e6).astype(np.float32)
    vertices=np.stack([x,y,z],axis=-1).reshape(-1,3)
    ry,rx=local_h.shape
    rows=np.arange(ry-1,dtype=np.uint32)[:,None]
    cols=np.arange(rx-1,dtype=np.uint32)[None,:]
    v0=(rows*rx+cols).ravel()
    v1=v0+1
    v2=v0+rx
    v3=v2+1
    faces=np.empty((2*len(v0),3),dtype=np.uint32)
    faces[::2]=np.stack((v0,v1,v2),axis=1)
    faces[1::2]=np.stack((v1,v3,v2),axis=1)
    mesh=mi.Mesh('ag_heightfield',vertex_count=len(vertices),face_count=len(faces))
    params=mi.traverse(mesh)
    params['vertex_positions']=mi.Float(vertices.ravel())
    params['faces']=mi.UInt(faces.ravel())
    params.update()
    return mesh, int(ry*rx), len(faces)


def emission_lookup(m,x_um,y_um):
    """Piecewise-constant source texels (not gamma encoded)."""
    ey,ex=m.emission.shape
    px=m.em_dx_m*1e6
    col=np.floor(x_um/px+ex/2).astype(np.int64)
    row=np.floor(y_um/px+ey/2).astype(np.int64)
    good=(col>=0)&(col<ex)&(row>=0)&(row<ey)
    vals=np.zeros(np.shape(x_um),dtype=np.float32)
    vals[good]=m.emission[row[good],col[good]]
    return vals


def pupil_nodes(nr,nt,na,n_out):
    for i in range(nr):
        r=(na/n_out)*math.sqrt((i+0.5)/nr)
        for j in range(nt):
            theta=2*math.pi*(j+(0.5 if i%2 else 0.0))/nt
            yield r*math.cos(theta),r*math.sin(theta)


def simulate_mitsuba(m):
    try:
        import mitsuba as mi
        import drjit as dr
    except ImportError as exc:
        raise RuntimeError('Mitsuba 3 is not installed: uv pip install mitsuba') from exc
    variants=mi.variants()
    if m.mitsuba_variant=='auto':
        if m.device=='cuda':
            choices=['cuda_ad_rgb','cuda_rgb','llvm_ad_rgb','llvm_rgb','scalar_rgb']
        else:
            choices=['llvm_ad_rgb','llvm_rgb','scalar_rgb','cuda_ad_rgb','cuda_rgb']
        variant=next((v for v in choices if v in variants),None)
        if variant is None: raise RuntimeError(f'Mitsuba has no compatible RGB variant: {variants}')
    else:
        variant=m.mitsuba_variant
    mi.set_variant(variant)
    print(f'  Mitsuba variant={variant}; measured height mesh stride={m.mesh_stride}',flush=True)
    mesh,nverts,nfaces=create_mitsuba_mesh(mi,m)
    scene=mi.load_dict({'type':'scene','ag_surface':mesh})
    print(f'  Mitsuba mesh: {nverts:,} vertices, {nfaces:,} triangles',flush=True)
    ny,nx=m.shape
    img=np.zeros((ny*nx,),dtype=np.float64)
    gap=m.gap_m*1e6
    dx=m.dx_m*1e6
    hmax=(float(np.max(m.h_m))-float(np.mean(m.h_m)))*1e6
    ztop=gap+hmax+4*dx
    clearance=ztop-gap
    eta=m.n_out/m.n_stack
    pupil=list(pupil_nodes(m.ray_radial,m.ray_angular,m.na,m.n_out))
    N=ny*nx
    for pidx,(tx,ty) in enumerate(pupil):
        airz=-math.sqrt(1-tx*tx-ty*ty)
        internalz=-math.sqrt(1-eta*eta*(tx*tx+ty*ty))
        for start in range(0,N,m.ray_chunk):
            end=min(N,start+m.ray_chunk)
            idx=np.arange(start,end,dtype=np.int64)
            yy=((idx//nx)-(ny-1)/2)*dx
            xx=((idx%nx)-(nx-1)/2)*dx
            xorigin=xx-gap*(eta*tx)/(-internalz)-clearance*tx/(-airz)
            yorigin=yy-gap*(eta*ty)/(-internalz)-clearance*ty/(-airz)
            org=mi.Point3f(mi.Float(xorigin.astype(np.float32)),
                           mi.Float(yorigin.astype(np.float32)),
                           mi.Float(np.full(len(idx),ztop,np.float32)))
            direction=mi.Vector3f(mi.Float(np.full(len(idx),tx,np.float32)),
                                  mi.Float(np.full(len(idx),ty,np.float32)),
                                  mi.Float(np.full(len(idx),airz,np.float32)))
            si=scene.ray_intersect(mi.Ray3f(org,direction),
                                   ray_flags=mi.RayFlags.Minimal,coherent=False)
            valid=np.asarray(si.is_valid(),dtype=bool)
            vals=np.zeros((len(idx),),dtype=np.float32)
            if valid.any():
                nxv=np.asarray(si.n.x)[valid]
                nyv=np.asarray(si.n.y)[valid]
                nzv=np.asarray(si.n.z)[valid]
                norm=np.sqrt(nxv**2+nyv**2+nzv**2)
                nxv=nxv/norm; nyv=nyv/norm; nzv=nzv/norm
                # Incoming air ray points toward negative z, normal points up.
                cosi=-(tx*nxv+ty*nyv+airz*nzv)
                valid_side=cosi>0
                cosi=np.maximum(cosi,0)
                ct2=1-eta**2*(1-cosi**2)
                refract=valid_side & (ct2>0)
                cost=np.sqrt(np.maximum(ct2,0))
                txs=eta*tx+(eta*cosi-cost)*nxv
                tys=eta*ty+(eta*cosi-cost)*nyv
                tzs=eta*airz+(eta*cosi-cost)*nzv
                refract &= tzs< -1e-6
                xhit=np.asarray(si.p.x)[valid]
                yhit=np.asarray(si.p.y)[valid]
                zhit=np.asarray(si.p.z)[valid]
                ztravel=np.divide(-zhit,tzs,out=np.zeros_like(tzs),where=refract)
                px=xhit+ztravel*txs
                py=yhit+ztravel*tys
                rad=emission_lookup(m,px,py)
                rs=((m.n_out*cosi-m.n_stack*cost)/(m.n_out*cosi+m.n_stack*cost+1e-15))**2
                rp=((m.n_stack*cosi-m.n_out*cost)/(m.n_stack*cosi+m.n_out*cost+1e-15))**2
                power=np.maximum(0,1-.5*(rs+rp))
                vals[valid]=np.where(refract,rad*power,0.)
            img[start:end]+=vals/len(pupil)
        if m.progress_every and ((pidx+1)%max(1,m.progress_every//2)==0 or pidx+1==len(pupil)):
            print(f'  Mitsuba pupil nodes: {pidx+1}/{len(pupil)}',flush=True)
    return img.reshape(ny,nx).astype(np.float32),N*len(pupil)


def grayscale_png(path,array,lo=None,hi=None):
    arr=np.asarray(array,dtype=np.float64)
    if lo is None: lo=float(np.quantile(arr,.001))
    if hi is None: hi=float(np.quantile(arr,.999))
    if hi<=lo: hi=lo+1e-30
    z=np.clip((arr-lo)/(hi-lo),0,1)
    Image.fromarray(np.rint(z*255).astype(np.uint8)).save(path)


def brightness_statistics(image, margin_px):
    """Direct population CV of all brightness pixels in a geometric ROI."""
    image=np.asarray(image)
    if image.ndim!=2 or not np.isfinite(image).all() or np.any(image<0):
        raise ValueError('Brightness must be a finite nonnegative 2D array')
    h,w=image.shape
    if margin_px<0 or 2*margin_px>=min(h,w):
        raise ValueError('ROI margin excludes the brightness image')
    mask=np.zeros((h,w),dtype=bool)
    mask[margin_px:h-margin_px,margin_px:w-margin_px]=True
    values=image[mask].astype(np.float64)
    mean=float(np.mean(values)); std=float(np.std(values,ddof=0))
    if mean<=0: raise ValueError('Mean brightness in the ROI must be positive')
    return {'sparkle_cv_percent':100*std/mean,'brightness_mean':mean,'brightness_std':std,
            'std_ddof':0,'num_roi_pixels':int(values.size),
            'roi_bounds_xyxy':[margin_px,margin_px,w-margin_px,h-margin_px]},mask


def collect_analysis_pixels(image, source_pitch_um, pixel_pitch_um, margin_um=0.):
    """Area-average intensity into full square detector pixels, centered in the ROI.

    Overlap weights integrate partial optical texels exactly for piecewise-constant
    intensity. Incomplete detector pixels at the ROI edges are excluded symmetrically.
    """
    image=np.asarray(image,dtype=np.float64)
    if image.ndim!=2 or not np.isfinite(image).all() or np.any(image<0):
        raise ValueError('Brightness must be a finite nonnegative 2D array')
    if (not np.isfinite([source_pitch_um,pixel_pitch_um,margin_um]).all()
            or min(source_pitch_um,pixel_pitch_um)<=0 or margin_um<0):
        raise ValueError('Pixel pitches must be positive and ROI margin nonnegative')

    def axis_weights(size):
        extent=size*source_pitch_um
        count=math.floor((extent-2*margin_um)/pixel_pitch_um+1e-10)
        if count<1: raise ValueError('ROI does not contain one full analysis pixel')
        start=(extent-count*pixel_pitch_um)/2
        edges=start+np.arange(count+1)*pixel_pitch_um
        fine=np.arange(size+1)*source_pitch_um
        overlap=np.maximum(0,np.minimum(edges[1:,None],fine[None,1:])
                           -np.maximum(edges[:-1,None],fine[None,:-1]))
        return overlap/pixel_pitch_um,start,extent-start

    wy,y0,y1=axis_weights(image.shape[0])
    wx,x0,x1=axis_weights(image.shape[1])
    collected=wy@image@wx.T
    h,w=image.shape
    return collected,{
        'analysis_grid':list(collected.shape),'analysis_pixel_pitch_um':pixel_pitch_um,
        'analysis_pixel_area_um2':pixel_pitch_um**2,
        'analysis_extent_um_xyxy':[x0-w*source_pitch_um/2,y0-h*source_pitch_um/2,
                                   x1-w*source_pitch_um/2,y1-h*source_pitch_um/2],
        'analysis_source_bounds_px_xyxy':[x0/source_pitch_um,y0/source_pitch_um,
                                         x1/source_pitch_um,y1/source_pitch_um],
        'effective_roi_margin_um_xy':[x0,y0],
        'analysis_resampling':'area-averaged linear brightness over square pixels',
        'metric_source_file':'analysis_brightness.npy'}


def postprocess(m,backend,brightness,out,elapsed,nmodes,opts):
    if brightness.shape!=m.shape: raise ValueError('Backend returned incorrect brightness shape')
    collected,geometry=collect_analysis_pixels(brightness,m.dx_m*1e6,
                                               m.display_pitch_m*1e6,m.roi_margin_m*1e6)
    metric,mask=brightness_statistics(collected,0)
    stats={
        'backend':backend,'physics_group':'scalar_wave' if backend in WAVE_BACKENDS else 'geometric_rays',
        'physical_model':'scalar ASM + thin phase screen + ideal NA imaging' if backend in WAVE_BACKENDS else 'Mitsuba mesh + Snell + pupil quadrature',
        'metric_definition':'100 * std(I_ROI, ddof=0) / mean(I_ROI)',
        'metric_domain':'area-averaged collected brightness pixels at analysis_pixel_pitch_um, including dark pixels',
        'brightness_units':'relative linear intensity','deterministic':True,
        'height_grid':list(m.shape),'emission_grid':list(m.emission.shape),
        'height_pitch_um':m.dx_m*1e6,'height_range_um':opts.height_range_um,
        'emission_pitch_um':m.em_dx_m*1e6,'gap_um':m.gap_m*1e6,
        'wavelength_nm':m.wavelength_m*1e9,'n_stack':m.n_stack,'n_out':m.n_out,'camera_na':m.na,
        'source_samples_per_axis':m.source_samples,
        'source_node_pitch_um':m.em_dx_m*1e6/m.source_samples if backend in WAVE_BACKENDS else None,
        'source_representation':'incoherent point quadrature' if backend in WAVE_BACKENDS else 'continuous emission texture',
        'field_width_um':m.shape[1]*m.dx_m*1e6,'field_height_um':m.shape[0]*m.dx_m*1e6,
        'pad_px':m.pad,'batch':m.batch,'roi_margin_um':m.roi_margin_m*1e6,'roi_margin_px':0,
        'torch_device_setting':m.device,'ray_radial':m.ray_radial,'ray_angular':m.ray_angular,
        'mesh_stride':m.mesh_stride,'mitsuba_variant':m.mitsuba_variant,
        'num_sources_or_rays':int(nmodes),'elapsed_seconds':round(elapsed,6),**geometry,**metric,
        'warnings':['Single effective wavelength and idealized camera optics; relative intensity units.',
                    'All full analysis pixels in the centered geometric ROI enter the statistic.']}
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'brightness.npy',brightness)
    np.save(out/'analysis_brightness.npy',collected)
    np.save(out/'roi_mask.npy',mask)
    grayscale_png(out/'brightness.png',brightness,0,max(float(np.max(brightness)),1e-30))
    grayscale_png(out/'analysis_brightness.png',collected,0,max(float(np.max(collected)),1e-30))
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def compare_results(results,root):
    metrics=[{'backend':name,'physics':s['physics_group'],'sparkle_cv_pct':s['sparkle_cv_percent'],
              'roi_pixels':s['num_roi_pixels'],'analysis_pixel_pitch_um':s['analysis_pixel_pitch_um'],
              'analysis_rows':s['analysis_grid'][0],'analysis_columns':s['analysis_grid'][1],
              'mean_brightness':s['brightness_mean'],
              'std_brightness':s['brightness_std'],'elapsed_s':s['elapsed_seconds']} for name,s in results.items()]
    if metrics:
        with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary={'metric_definition':'100 * std(I_ROI, ddof=0) / mean(I_ROI)',
             'backends':metrics,'important':'Statistics use area-averaged floating-point brightness pixels at the requested physical pixel size, before PNG display scaling.'}
    (root/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for row in metrics: print(f'{row["backend"]}: brightness CV = {row["sparkle_cv_pct"]:.6f}%, solver = {row["elapsed_s"]:.3f}s')


def run(argv=None):
    run_start=time.perf_counter()
    argv=list(sys.argv[1:] if argv is None else argv)
    a=parse_args(argv)
    root=Path(a.out).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    run_meta=capture_run(a,argv,Path(__file__))
    a._run_id=run_meta['run_id']
    run_meta['results_directory']=str(root)
    run_meta['comparison_written']=False
    backends=list(ALL_BACKENDS) if a.backend=='all' else [a.backend]
    run_meta['backend_runs']={name:{'status':'not_run'} for name in backends}
    summary={}
    errors={}
    try:
        m=model_from_args(a)
        src=enumerate_sources(m)
        run_meta['effective_model']=model_metadata(m,len(src))
        H,W=m.padded_shape
        print(f'Height: {m.shape[0]}x{m.shape[1]} at {m.dx_m*1e6:g}um, '
              f'emission: {m.emission.shape[0]}x{m.emission.shape[1]} at {m.em_dx_m*1e6:g}um')
        print(f'Wave FFT: {H}x{W}; source modes {len(src)}; '
              f'one complex64 field: {H*W*8/2**20:.1f} MiB')
        print(f'Mitsuba rays: {m.shape[0]*m.shape[1]*m.ray_radial*m.ray_angular:,}; '
              f'mesh stride {m.mesh_stride}')
        if a.dry_run:
            run_meta['status']='dry_run'
            return {}
        def run_backend(name):
            print(f'\n=== {name} ===',flush=True)
            started_at=utc_now()
            start=time.perf_counter()
            try:
                if name in ('torch','torchoptics'):
                    ag,nm=simulate_torch_wave(m,name,a.max_sources)
                elif name=='chromatix':
                    ag,nm=simulate_chromatix(m,a.max_sources)
                elif name=='mitsuba':
                    ag,nm=simulate_mitsuba(m)
                else: raise AssertionError(name)
                elapsed=time.perf_counter()-start
                save_start=time.perf_counter()
                stats=postprocess(m,name,ag,root/name,elapsed,nm,a)
                stats.update({'run_id':run_meta['run_id'],'started_at_utc':started_at,
                              'finished_at_utc':utc_now(),
                              'postprocess_elapsed_seconds':round(time.perf_counter()-save_start,6),
                              'execution_context':execution_context(name,m)})
                (root/name/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
                summary[name]=stats
                run_meta['backend_runs'][name]={'status':'complete','solver_elapsed_seconds':elapsed}
                print(f'  Sparkle CV: {stats["sparkle_cv_percent"]:.5f}%')
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc)}
                errors[name]=error
                missing_dependency=(isinstance(exc,RuntimeError) and
                                    ('not installed' in str(exc) or 'missing:' in str(exc)))
                skipped=a.backend=='all' and a.skip_unavailable and missing_dependency
                run_meta['backend_runs'][name]={'status':'skipped' if skipped else 'failed',
                    'error':error,'elapsed_before_error_seconds':time.perf_counter()-start}
                if skipped:
                    print(f'SKIPPED {name}: {exc}',file=sys.stderr)
                    return
                print(f'FAILED {name}: {exc}',file=sys.stderr)
                raise
        if a.backend_workers==1 or len(backends)==1:
            for name in backends: run_backend(name)
        else:
            with ThreadPoolExecutor(max_workers=a.backend_workers) as executor:
                futures=[executor.submit(run_backend,name) for name in backends]
                for future in as_completed(futures): future.result()
        summary={name:summary[name] for name in backends if name in summary}
        if not summary:
            raise RuntimeError('No backend completed successfully; see errors above')
        compare_results(summary,root)
        run_meta['comparison_written']=True
        run_meta['status']='complete'
        return summary
    except Exception as exc:
        run_meta['status']='failed'
        run_meta['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        if run_meta['status']=='running':
            run_meta['status']='interrupted'
        run_meta['finished_at_utc']=utc_now()
        run_meta['wall_elapsed_seconds']=round(time.perf_counter()-run_start,6)
        (root/'run.json').write_text(json.dumps(run_meta,ensure_ascii=False,indent=2),encoding='utf-8')
        errfile=root/'errors.json'
        if errors:
            errfile.write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
        elif errfile.exists():
            errfile.unlink()
        report=write_html_report(root)
        print('HTML report:',report)


if __name__=='__main__':
    run()
