from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "emotion_utils",
        ["emotion_utils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["/O2"] if __import__('sys').platform == 'win32' else ["-O3", "-march=native"],
    )
]

setup(
    name="emotion_utils",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
    zip_safe=False,
)

