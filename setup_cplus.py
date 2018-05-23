from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
           "var_elim",                                # the extension name
           sources=["var_elim.pyx", "variable_elimination.cpp"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11","-fopenmp"], 
           extra_link_args=["-std=c++11","-fopenmp"],
           include_dirs=[numpy.get_include()],
      )))

# cd /home/marcobb8/Dropbox/Phd/PTrees/code2/
# python setup_cplus.py build_ext --inplace