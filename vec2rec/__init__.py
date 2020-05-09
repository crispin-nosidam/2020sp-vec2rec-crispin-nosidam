from setuptools_scm import get_version

# Call setuptools_scm to include version
try:
    __version__ = get_version()
except:
    pass
