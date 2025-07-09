# nbody_utils/__init__.py

__version__ = "0.1.0"

try:
    import jax.numpy 
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .assign_util_jax import assign
    from .power_util_jax import Measure_Pk, Measure_spectra_FFT, rfftn_kvec
else:
    from .assign_util import assign
    from .power_util import Measure_Pk, Measure_spectra_FFT, rfftn_kvec

__all__ = [
    "rfftn_kvec",
    "Measure_spectra_FFT",
    "Measure_Pk",
    "assign",
]
