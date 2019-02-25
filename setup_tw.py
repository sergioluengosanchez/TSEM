from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="tree_width",
    ext_modules=cythonize("tree_width.pyx"),
    include_dirs=[numpy.get_include()]
)
