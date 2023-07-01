from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pyfit',
    version='0.0.1',
    packages=find_packages(),
    install_requires=required,
    author='xflynx25',
    author_email='JFlynn250@outlook.com',
    description='Pyfit, your ml toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/xflynx25/pyfit',
    project_urls = {
        "Bug Tracker": "https://github.com/xflynx25/pyfit/issues"
    },
    license='GNU',
    #packages=['pyfit'],
    #install_requires=[],
)

from setuptools import setup, find_packages