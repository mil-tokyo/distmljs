# DistML.js distributed training server

Server-side code is implemented as Python library.

# Setup

Python 3.8+

```bash
pip install -r requirements.txt
pip install -e .
```

How to run sample: see `sample/*/README.md`

# Build for distribution

Prerequisites:

```bash
pip install build
```

Build:

```bash
python -m build --wheel
```

`dist/distmljs-<version>-py3-none-any.whl` will be generated. The user runs `pip install /path/to/distmljs-<version>-py3-none-any.whl` to install DistML.js along with required dependencies (numpy, etc.).
