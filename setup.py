import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyfit',
    version='0.0.1',
    author='xflynx25',
    author_email='JFlynn250@outlook.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/xflynx25/pyfit',
    project_urls = {
        "Bug Tracker": "https://github.com/xflynx25/pyfit/issues"
    },
    license='GNU',
    packages=['pyfit'],
    install_requires=[],
)