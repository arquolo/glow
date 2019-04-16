#!/bin/python3
import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='ort',
    version='0.0.3',
    url="https://github.com/arquolo/ort",
    author='Paul Maevskikh',
    author_email='arquolo@gmail.com',
    description='Optimized RouTines for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['wrapt'],
)
