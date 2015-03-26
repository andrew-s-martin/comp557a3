# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:03:21 2015

@author: Fahim
"""

from distutils.core import setup
from Cython.Build import cythonize

# add any additional modules that need to be cythonized
source_files = ["Intersectable.py", "HelperClasses.py", 
               "Ray.py", "SceneParser.py"]
setup(
    ext_modules = cythonize(source_files)
)

