"""Report integrity: missing data, stale results, escaping and failed runs."""
import json
import base64
import io
import re
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

from test_smoke import ReportDataParser

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reporting import execution_context, image_metadata, write_html_report


def payload(path):
    parser=ReportDataParser()
    parser.feed(path.read_text(encoding='utf-8'))
    return json.loads(''.join(parser.payload))


class HtmlReportingTest(unittest.TestCase):
    def test_remote_chromatix_context_does_not_require_parent_jax(self):
        from types import SimpleNamespace
        remote = {'backend': 'chromatix', 'runtime': 'wsl', 'devices': ['cuda:0'],
                  'environment': {'executable': '/home/user/venv/bin/python'}}
        model = SimpleNamespace(_chromatix_execution_context=remote)
        result = execution_context('chromatix', model)
        self.assertEqual(result, remote)
        self.assertIsNot(result, remote)

    def test_report_shows_detector_pixels_and_keeps_both_float_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder=root/'torch'; folder.mkdir()
            fine=np.arange(16,dtype=np.float32).reshape(4,4)
            collected=fine.astype(np.float64).reshape(2,2,2,2).mean(axis=(1,3))
            np.save(folder/'brightness.npy',fine); np.save(folder/'analysis_brightness.npy',collected)
            Image.fromarray(np.arange(16,dtype=np.uint8).reshape(4,4)).save(folder/'brightness.png')
            Image.fromarray(np.array([[0,80],[160,255]],dtype=np.uint8)).save(folder/'analysis_brightness.png')
            stats={'backend':'torch','sparkle_cv_percent':50,'analysis_grid':[2,2],
                   'analysis_pixel_pitch_um':50}
            (folder/'stats.json').write_text(json.dumps(stats),encoding='utf-8')
            text=write_html_report(root).read_text(encoding='utf-8')
            self.assertIn('每像素 50 × 50 µm',text)
            self.assertIn('torch-analysis_brightness.npy',text)
            self.assertIn('torch-brightness.npy',text)
            encoded=re.search(r'<img[^>]+src="data:image/png;base64,([^"]+)"',text).group(1)
            displayed=Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
            np.testing.assert_array_equal(np.asarray(displayed),[[0,80],[160,255]])

    def test_input_gallery_preserves_originals_is_visible_and_deduplicated(self):
        class GalleryParser(HTMLParser):
            def __init__(self):
                super().__init__(); self.inside=False; self.details=0
                self.images=[]; self.downloads=[]; self.gallery_before_results=False
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if tag=='details': self.details+=1
                if tag=='section' and attrs.get('id')=='input-images':
                    self.inside=True; self.gallery_before_results=True
                if tag=='img' and self.inside: self.images.append((self.details,attrs['src']))
                if tag=='a' and self.inside and 'download' in attrs:
                    self.downloads.append((attrs['download'],attrs['href']))
                if tag=='section' and attrs.get('id')=='results':
                    assert self.gallery_before_results
            def handle_endtag(self,tag):
                if tag=='section' and self.inside: self.inside=False
                if tag=='details': self.details-=1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            height=root/'height.png'; emission=root/'emission.png'
            Image.fromarray((np.arange(64,dtype=np.uint16)*1024).reshape(8,8)).save(height)
            Image.fromarray(np.full((4,4,3),255,dtype=np.uint8)).save(emission)
            inputs={'height':image_metadata(height,2),'emission':image_metadata(emission,4)}
            for name in ('first','second'):
                (root/name).mkdir()
                (root/name/'run.json').write_text(json.dumps({'run_id':name,'status':'dry_run',
                    'parameters':{'height_range_um':.5,'emission_channel':'luma','gamma_decode':False},
                    'input_files':inputs,'backend_runs':{'torch':{'status':'not_run'}}}),encoding='utf-8')
            result=write_html_report(root,recursive=True)
            parser=GalleryParser(); parser.feed(result.read_text(encoding='utf-8'))
            self.assertEqual(len(parser.images),2)
            self.assertTrue(all(depth==0 for depth,_ in parser.images))
            files={name:base64.b64decode(url.split(',',1)[1]) for name,url in parser.downloads}
            self.assertEqual(files['height.png'],height.read_bytes())
            self.assertEqual(files['emission.png'],emission.read_bytes())
            self.assertEqual(files['height.png'][24],16)
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(files['height.png']))),
                                          (np.arange(64,dtype=np.uint16)*1024).reshape(8,8))
            self.assertEqual(len(payload(result)['input_images']),2)
            self.assertTrue(all(len(item['uses'])==2 for item in payload(result)['input_images']))
            Image.fromarray(np.zeros((8,8),dtype=np.uint16)).save(height)
            data=payload(write_html_report(root,recursive=True))
            embedded={item['role']:item['original_embedded'] for item in data['input_images']}
            self.assertEqual(embedded,{'height':False,'emission':True})

    def test_escape_and_missing_values_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'torch').mkdir()
            dangerous='</script><img src=x onerror="alert(1)">'
            stats={'backend':'torch','physics_group':'scalar_wave',
                   'sparkle_cv_percent':1.25,'warnings':[dangerous]}
            (root/'torch/stats.json').write_text(json.dumps(stats),encoding='utf-8')
            result=write_html_report(root,title=dangerous)
            text=result.read_text(encoding='utf-8')
            data=payload(result)
            self.assertEqual(data['groups'][0]['records'][0]['stats'],stats)
            self.assertNotIn('<img src=x onerror=',text)
            self.assertEqual(text.count('<script '),1)
            self.assertNotIn('elapsed_seconds',data['groups'][0]['records'][0]['stats'])
            self.assertEqual(data['groups'][0]['run']['parameter_provenance'],'legacy_partial')

    def test_stale_and_skipped_outputs_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for backend in ('torch','mitsuba'):
                (root/backend).mkdir()
                (root/backend/'stats.json').write_text(json.dumps({
                    'backend':backend,'run_id':'old','sparkle_cv_percent':999}),encoding='utf-8')
            (root/'run.json').write_text(json.dumps({'run_id':'new','status':'failed',
                'backend_runs':{'torch':{'status':'complete'},'mitsuba':{'status':'skipped'}}}),encoding='utf-8')
            data=payload(write_html_report(root))
            records=data['groups'][0]['records']
            self.assertEqual([record['status'] for record in records],['failed','skipped'])
            self.assertTrue(all(record['stats'] is None for record in records))

    def test_failed_solver_still_writes_input_snapshot_and_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            command=[sys.executable,str(ROOT/'scripts/run_demo.py'),'--backend','torch',
                     '--device','cpu','--out',str(root),'--extra','--max-sources','1']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            data=payload(root/'report.html')
            group=data['groups'][0]
            self.assertEqual(group['run']['status'],'failed')
            self.assertEqual(group['run']['parameters']['max_sources'],1)
            self.assertEqual(group['records'][0]['status'],'failed')
            self.assertIsNone(group['records'][0]['stats'])


if __name__=='__main__': unittest.main()
