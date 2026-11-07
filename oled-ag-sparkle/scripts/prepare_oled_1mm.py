#!/usr/bin/env python3
"""Prepare the existing height and sourced Diamond Pixel image on a 1 mm field."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def prepare(height_pitch_um=1.):
    if height_pitch_um not in (1.,2.): raise ValueError('Bundled height pitch must be 1 or 2 um')
    if height_pitch_um==1.:
        base=ROOT/'assets/oled-1mm'
        if not (base/'heightfield.png').exists(): prepare(2.)
        output=ROOT/'assets/oled-1mm-height1um'
        output.mkdir(parents=True,exist_ok=True)
        source_height=base/'heightfield.png'
        original=np.asarray(Image.open(source_height),dtype=np.float32)
        height=np.asarray(Image.fromarray(original).resize((1000,1000),Image.Resampling.BILINEAR))
        Image.fromarray(np.rint(height).clip(0,65535).astype(np.uint16)).save(output/'heightfield.png')
        (output/'heightfield.source.json').write_text(json.dumps({
            'source':'existing assets/oled-1mm/heightfield.png',
            'original_sha256':hashlib.sha256(source_height.read_bytes()).hexdigest(),
            'original_size_px':[500,500],'resized_to_px':[1000,1000],
            'resampling':'bilinear interpolation of existing 16-bit height values; no new measured detail',
            'assigned_field_um':[1000,1000],'assigned_texel_um':1,
            'height_range_um':0.5},indent=2),encoding='utf-8')
        return output
    output=ROOT/'assets/oled-1mm'
    output.mkdir(parents=True,exist_ok=True)
    source_height=ROOT/'assets/demo_quick/heightfield.png'
    original_height=np.asarray(Image.open(source_height),dtype=np.float32)
    # Interpolate height values, retaining 16-bit PNG encoding and the 0.5 um range.
    height=np.asarray(Image.fromarray(original_height).resize((500,500),Image.Resampling.BILINEAR))
    Image.fromarray(np.rint(height).clip(0,65535).astype(np.uint16)).save(output/'heightfield.png')
    (output/'heightfield.source.json').write_text(json.dumps({
        'source':'existing synthetic demo_quick/heightfield.png',
        'original_sha256':hashlib.sha256(source_height.read_bytes()).hexdigest(),
        'original_size_px':[128,128],'resized_to_px':[500,500],
        'resampling':'bilinear interpolation of 16-bit height values',
        'assigned_field_um':[1000,1000],'assigned_texel_um':2,
        'height_range_um':0.5},indent=2),encoding='utf-8')
    provenance=json.loads((ROOT/'assets/oled-diamond/emission.source.json').read_text(encoding='utf-8'))
    with Image.open(ROOT/'assets/oled-diamond/samsung-diamond-reference.png') as original:
        original.crop(tuple(provenance['crop_xyxy'])).resize((500,500),Image.Resampling.BOX).save(output/'emission.png')
    provenance.update(resized_to_px=[500,500],assigned_emission_texel_um=2,
                      assigned_field_um=[1000,1000])
    (output/'emission.source.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return output


if __name__=='__main__': print(prepare())
