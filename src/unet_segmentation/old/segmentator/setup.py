from setuptools import setup
from Cython.Build import cythonize

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
setup(
    ext_modules=cythonize("builder/set_class.pyx", **ext_options)
)
