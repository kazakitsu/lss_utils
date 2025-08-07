from setuptools import setup, find_packages

setup(
    name="lss_utils",
    version="0.1.0",
    description="Utility routines for analyzing N-body data",
    author="Kazuyuki Akitsu",
    url="https://github.com/kazakitsu/lss_utils",
    package_dir={"": "lss_utils"},
    packages=find_packages(where='lss_utils'),
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
    },
    python_requires=">=3.7",
)
