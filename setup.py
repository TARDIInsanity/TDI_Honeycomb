# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:43:00 2022

@author: github.com/TARDIInsanity
"""

from setuptools import setup

__version__ = "2"

setup(
      name = "TDI_Honeycomb",
      version = __version__,
      
      url = "https://github.com/TARDIInsanity/TDI_Honeycomb",
      author = "TARDIInsanity",
      
      py_modules = ["suki_02"+i for i in ("", "_lexer", "_parser_03")],
)
