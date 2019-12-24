from setuptools import setup
from distutils.sysconfig import get_python_lib
import glob


setup(
    name = "PyFastTsne",
    package_dir = {'': 'PyFastTsne'},
    data_files = [(get_python_lib(), glob.glob('PyFastTsne/*.so'))
                 ],
    author = 'BalanSky',
    description = 'FastTsne Python Wraper.',
    license = 'Apache',
    keywords = 'TSNE',
    url = 'https://github.com/balansky/Tsne',
    zip_safe = False,
)