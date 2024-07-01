# DistML.js distributed training server

Server-side code is implemented as Python library.

# Setup

Python 3.8+

```
pip install -r requirements.txt
python setup.py develop
```

How to run sample: see `samples/*/README.md`

# Build for distribution

```
python setup.py bdist_wheel
```

`dist/distmljs-<version>-py3-none-any.whl` will be generated. The user runs `pip install /path/to/distmljs-<version>-py3-none-any.whl` to install DistML.js along with required dependencies (numpy, etc.).
