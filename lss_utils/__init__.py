# lss_utils/__init__.py

__version__ = "0.3.0"

try:
    import jax.numpy 
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .assign_util_jax import Mesh_Assignment
    from .spectra_util_jax import Measure_Pk, Measure_spectra_FFT, Measure_Xi_FFT
    from .tensor_utils_jax import kaxes_1d, project_rank2_helicity
else:
    from .assign_util import Mesh_Assignment
    from .spectra_util import Measure_Pk, Measure_spectra_FFT
    from .tensor_utils import kaxes_1d, project_rank2_helicity

__all__ = [
    "Mesh_Assignment",
    "Measure_Pk",
    "Measure_spectra_FFT",
    "kaxes_1d",
    "project_rank2_helicity",
]
