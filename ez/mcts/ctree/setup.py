from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cytree",
        ["cytree.pyx"],
        extra_compile_args=["-O3", "-Wall", "-Wextra"],
        include_dirs=[np.get_include()]
    )
]

compiler_directives = {
    "language_level": "3"
}

setup(
    name='Cython Tree Module',
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives), include_dirs=[np.get_include()]
)

