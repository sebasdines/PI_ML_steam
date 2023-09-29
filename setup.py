from setuptools import setup
setup(
    name='PI_ML_steam',
    version='1.0.0',
    description='Project Machine Learning',
    author='Sebastian DI Nesta',
    author_email='your.email@example.com',
    packages=['my_project'],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'fastapi',
        'fastparquet',
        'joblib',
        'pyarrow',
        'sklearn',
        'uvicorn'
    ],
)