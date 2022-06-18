from setuptools import setup, find_packages
from src import GraphConstruction

requirements = open('./requirements.txt', 'r').readlines()

setup(
    name='GraphConstruction',
    version=GraphConstruction.__version__,
    description='GraphConstruction Python package',
    author=GraphConstruction.__author__,
    author_email=GraphConstruction.__contact__,
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=requirements,
    extras_require={
        "cknn": ["git+https://github.com/chlorochrule/cknn/blob/master/cknn/cknn.py"],
        "distanceclosure": ["git+https://github.com/tarikaltuncu/distanceclosure"],
    }
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)