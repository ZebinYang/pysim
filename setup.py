from setuptools import setup

setup(name='pysim',
      version='0.1',
      description='Single index model based on first and second order Stein method',
      url='https://github.com/ZebinYang/pysim',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['pysim'],
      install_requires=[
          'matplotlib','patsy', 'numpy', 'sklearn', 'rpy2'],
      zip_safe=False)
