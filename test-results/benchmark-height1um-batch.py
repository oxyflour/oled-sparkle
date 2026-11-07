"""Short throughput probe on the real 1um grid; never used as report results."""
from pathlib import Path
import sys,time,json
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'oled-ag-sparkle/scripts'))
import sparkle_multibackend as s
a=s.parse_args(['--height',str(ROOT/'oled-ag-sparkle/assets/oled-1mm-height1um/heightfield.png'),
 '--height-pitch-um','1','--height-range-um','.5',
 '--emission',str(ROOT/'oled-ag-sparkle/assets/oled-1mm/emission.png'),'--emission-pitch-um','2',
 '--gamma-decode','--pad-px','140','--device','cpu','--progress-every','0'])
m=s.model_from_args(a)
source=s.enumerate_sources(m)[:512]
s.enumerate_sources=lambda model:source
result=[]
for batch in (1,4,16):
 start=time.perf_counter()
 image,count=s.simulate_chromatix(replace(m,batch=batch),100000)
 elapsed=time.perf_counter()-start
 result.append({'batch':batch,'seconds':elapsed,'probe_sources':count})
 print(result[-1],flush=True)
(ROOT/'test-results/height1um-batch-probe.json').write_text(json.dumps(result,indent=2))
