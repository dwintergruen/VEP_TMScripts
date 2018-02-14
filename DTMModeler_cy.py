from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize






extensions = [Extension("VEP_TMScripts.DTMModeler", ["DTMModeler.py"]),
              Extension("ads.__init__", ["/usr/local/lib/python3.6/site-packages/ads/__init__.py"]),
              Extension("ads.*", ["/usr/local/lib/python3.6/site-packages/ads/*.py"]),
              Extension("werkzeug.__init__", ["/usr/local/lib/python3.6/site-packages/werkzeug/__init__.py"]),
              Extension("werkzeug.*", ["/usr/local/lib/python3.6/site-packages/werkzeug/*.py"]),
              Extension("sklearn.__init__", ["/usr/local/lib/python3.6/site-packages/werkzeug/__init__.py"]),
              Extension("sklearn.*", ["/usr/local/lib/python3.6/site-packages/sklearn/*.py"]),
              Extension("sklearn.*.*", ["/usr/local/lib/python3.6/site-packages/sklearn/*/*.py"]),
              Extension("pandas.*", ["/usr/local/lib/python3.6/site-packages/pandas/*.py"]),
              Extension("pandas.*.*", ["/usr/local/lib/python3.6/site-packages/pandas/*/*.py"])]

setup(
    name = "VEP_TMScripts",
    ext_modules = cythonize(extensions)
)