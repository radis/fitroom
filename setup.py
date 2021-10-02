from setuptools import setup, find_packages

setup(
    name='fitroom',
    version='0.1',
    url='https://github.com/radis/fitroom',
    author='Erwan Pannier',
    author_email='erwan.pannier@gmail.com',
    description='Interactive Tools for Multi-dimensional fitting of Absorption and Emission Spectra',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1', 'radis >= 0.10.3'],
)
