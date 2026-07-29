# Runs every WGSL shader generator under tools/.
#
# Usage:
#   python tools/generate_webgpu_shaders.py
#   node tools/compile_webgpu_shader.js

import glob
import os
import runpy

tools_dir = os.path.dirname(os.path.abspath(__file__))

scripts = sorted(
    glob.glob(os.path.join(tools_dir, "generate_webgputensor_wgsl_*.py"))
)
if not scripts:
    raise SystemExit("no generator script was found")

for script in scripts:
    print(os.path.basename(script))
    runpy.run_path(script, run_name="__main__")
