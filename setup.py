from setuptools import setup, find_packages

setup(
    name="nbody_utils",
    version="0.1.0",
    description="Utility routines for analyzing N-body data",
    author="Your Name",
    url="https://github.com/kazakitsu/nbody_utils",
    packages=find_packages(),
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
