from setuptools import setup, find_packages

setup(
    name='tac_lib',        # Name of the package
    version='0.2',                   # Version number
    author='Vishwanath1999',              # Author name
    author_email='viswasankar98@gmail.com',  # Author email
    description='TACBC is a Python library for simulating tiled aperture coherent beam combining.',  # Short description
    long_description=open('README.md').read(),  # Long description from README file
    long_description_content_type='text/markdown',  # Format of the long description
    url='https://github.com/Vishwanath1999/tac_lib',  # URL to the project's homepage
    packages=find_packages(),        # Packages to include in the distribution
    install_requires=[               # List of dependencies #TODO #FIXME #CHECK
        # 'numba>=0.58.0',
        # 'numpy>=1.24.1',
        # 'opencv-python>=4.8.0.76',
        # 'scipy>=1.11.2',
    ],
    classifiers=[                    # Additional metadata about the package
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.10',         # Python version requirement
)
