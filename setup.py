from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
try:
    # Try to format our PyPi page as rst so it displays properly
    import pypandoc
    with open ('README.md', 'rb') as read_file:
        readme_text = read_file.readlines()
    # Change our README for pypi so we can get analytics tracking information for that separately
    readme_text = [row.decode() for row in readme_text]
    readme_text[-1] = "[![Analytics](https://ga-beacon.appspot.com/UA-58170643-5/concordia/pypi)](https://github.com/igrigorik/ga-beacon)"

    long_description = pypandoc.convert(''.join(readme_text), 'rst', format='md')
except ImportError:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('pypandoc (and possibly pandoc) are not installed. This means the PyPi package info will be formatted as .md instead of .rst. If you are encountering this before uploading a PyPi distribution, please install these')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='concordia',

    version=open("concordia/_version.py").readlines()[-1].split()[-1].strip("\"'"),

    description='Automated monitoring of machine learning models in production. Tracks and finds discrepancies in features, predictions, and labels',
    long_description=long_description,

    url='https://github.com/ClimbsRocks/Concordia',

    author='Preston Parry',
    author_email='ClimbsBytes@gmail.com',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords=['machine learning', 'data science', 'automated machine learning', 'deploying', 'machine learning in production', 'productionizing machine learning', 'tracking', 'feature discrepancies', 'train/serve skew', 'train serve skew', 'train-serve skew', 'model accuracy', 'alerts', 'monitoring', 'production ready', 'test coverage'],

    packages=['concordia'],

    install_requires=[
        'auto_ml>=2.9.4',
        'dill>=0.2.3, <0.3',
        'pymongo>3.0, <4.0',
        'redis>2.0, <3.0'
    ],

    test_suite='nose.collector',
    tests_require=['nose', 'coveralls']
)
