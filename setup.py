from setuptools import setup, find_packages

setup(
    name="lss_utils",
    version="0.2.0",
    description="Utility routines for analyzing N-body data",
    author="Kazuyuki Akitsu",
    url="https://github.com/kazakitsu/lss_utils",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.3",
            "jaxlib>=0.4.3",
        ],
    },
    python_requires=">=3.10",
)
