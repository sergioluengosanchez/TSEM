from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
	name = "data_type",
	ext_modules= cythonize("data_type.pyx"),
    include_dirs=[numpy.get_include()]
)