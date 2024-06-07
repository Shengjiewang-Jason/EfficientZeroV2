from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# ext = Extension(name='ctree', sources=['ctree.pxd'])
# ext = Extension(name='cytree', sources=['cytree.pyx'],
#                 # extra_compile_args=['-O3'],
#                 language="c++")
setup(ext_modules=cythonize('cytree.pyx'), extra_compile_args=['-O3'], include_dirs=[np.get_include()])

