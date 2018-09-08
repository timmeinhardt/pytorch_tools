from setuptools import setup, find_packages

setup(name='pytorch_tools',
      packages=['pytorch_tools'],
      package_dir={'':'src'},
      version='0.0.1',
      install_requires=[],
      package_data={'': ['config/*.yaml']},)
