import sys
import os
import pysim
sys.path.insert(0, os.path.abspath('../../pysim/'))
sys.path.insert(0, os.path.abspath('../_ext'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

source_suffix = '.rst'
master_doc = 'index'
project = u'pysim'

__version__ = pysim.__version__
__author__ = u'Zebin Yang' 
copyright = '2020, Zebin Yang'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
autoclass_content = "both"
napoleon_include_private_with_doc = True

html_show_sourcelink = True
html_context = {
  'display_github': True,
  'github_user': 'ZebinYang',
  'github_repo': 'pysim',
  'github_version': 'master/docs/source/'
}