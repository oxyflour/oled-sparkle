"""Current entry point: 1mm field, 1um height grid, 50um brightness pixel CV."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root.parent/'oled-ag-sparkle/scripts/run_oled_demo.py'),'--height-surface','original','--out',str(root/'oled-1mm-height1um-50um')],check=True)
