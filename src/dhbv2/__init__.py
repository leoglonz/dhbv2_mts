from dhbv2._version import __version__
from dhbv2.pet import calc_hourly_hargreaves_pet
from dhbv2.bmi import DeltaModelBmi
from dhbv2.bmi_mts import DeltaModelBmi as MtsDeltaModelBmi

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    "calc_hourly_hargreaves_pet",
    "DeltaModelBmi",
    "MtsDeltaModelBmi",
]
