from setuptools import setup, find_packages

setup(
    name="kakiage",
    packages=find_packages(),
    version="1.0.0",
    install_requires=["numpy", "fastapi", "uvicorn[standard]"]
)
