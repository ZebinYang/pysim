from setuptools import setup

setup(name='pysim',
      version='1.0',
      description='Single index model based on first and second order Stein method',
      url='https://github.com/ZebinYang/pysim',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['pysim', 'pysim.splines'],
      install_requires=[
          'matplotlib', 'numpy', 'sklearn', 'pygam'],
      zip_safe=False)
