from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='plldesigner',
    
    version='0.1.2',
    
    py_modules=['plldesigner.sdmod', 'plldesigner.pll',
                'plldesigner.pnoise'],
    
    description='Python toolkit for PLLs design and phase noise analysis',
    
    #Author details
    author='Juan F. Osorio',
    author_email='jfosorio@gmail.com',
    
    #Project web page
    url='https://github.com/jfosorio/plldesigner',
    
    
    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],

     )
