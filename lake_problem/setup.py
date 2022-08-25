from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [Extension("lakemodel", ["dps.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
