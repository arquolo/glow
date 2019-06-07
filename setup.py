#!/bin/python3

import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='glow',
    version='0.3',
    url='https://github.com/arquolo/glow',
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
    packages=setuptools.find_namespace_packages(),
    python_requires='>=3.6',
    install_requires=[
        "dataclasses ; python_version<'3.7'",
        'wrapt',
    ],
    extras_require={
        'vision': [
            'numpy>=1.15',
            'numba',
            'pyaudio',
            'soundfile',
        ],
        'nn': [
            'graphviz',
            'numpy>=1.15',
            'opencv-python>=4.0',
            'py3nvml',
            'pytorch>=1.1',
        ],
    },
)
