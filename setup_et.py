from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="elimination_tree",
    ext_modules=cythonize("elimination_tree.pyx"),
    include_dirs=[numpy.get_include()]
)