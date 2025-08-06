# lss_utils/__init__.py

__version__ = "0.1.0"

try:
    import jax.numpy 
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .assign_util_jax import Mesh_Assignment
    from .power_util_jax import rfftn_kvec, Measure_Pk, Measure_spectra_FFT
else:
    from .assign_util import Mesh_Assignment
    from .power_util import rfftn_kvec, Measure_Pk, Measure_spectra_FFT

__all__ = [
    "Mesh_Assignment",
    "rfftn_kvec",
    "Measure_Pk",
    "Measure_spectra_FFT",
]
