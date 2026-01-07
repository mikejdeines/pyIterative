from setuptools import setup, find_packages
setup(
    name='pyIterative',
    version='0.1.0',
    author='Michael J. Deines',
    author_email='michaeljdeines@gmail.com',
    description='A package for iterative clustering based on CONCORD and weighted t-tests.',
    url='https://github.com/mikejdeines/pyIterative',
    packages=find_packages(include=['pyiterative*']),
    install_requires=[
        'scanpy',
        'pandas<2.0.0',
        'igraph',
        'leidenalg',
        'scvi-tools>=0.18.1',
        'numpy>=1.24.4,<2.0.0',
        'scipy>=1.7.0',
        'statsmodels>=0.13.0',
        'concord-sc',
        'torch>=2.0.0',
        'pynndescent>=0.5.6',
        'tqdm>=4.60.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: macOS or Linux',
    ],
    python_requires='>=3.6',
)