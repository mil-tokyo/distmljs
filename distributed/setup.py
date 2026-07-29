from setuptools import setup, find_packages

setup(
    name="distmljs",
    packages=find_packages(),
    version="1.0.0",
    python_requires=">=3.8",
    install_requires=["numpy", "fastapi", "uvicorn[standard]"],
)
