from setuptools import setup, find_packages

setup(
    name='quadrotor-game-solver',
    version='0.1.0',
    author='MPC_2p0s1',
    description='3-D Hexner game solver with nonlinear 6-DoF quadrotor dynamics (SQP inner layer).',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch>=2.0',
        'pytest',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)