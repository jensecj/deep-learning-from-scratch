import sys
from setuptools import setup, find_packages

version = sys.version_info[:2]
if version < (3, 7):
    print('dlfs requires Python version 3.7 or later' + ' ({}.{} detected).'.format(*version))
    sys.exit(-1)

setup(name='dlfs',
      version='0.1.0',
      description='Simple Deep Learning Implementation',
      url='http://github.com/jensecj/deep-learning-from-scratch',
      author='Jens Christian Jensen',
      author_email='jensecj@gmail.com',
      packages=['dlfs'],
      zip_safe=False)
