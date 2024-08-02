from setuptools import setup, find_packages

setup(
    name="finlm",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Ralf Kellner",
    author_email="ralf.kellner@uni-passau.de",
    description="Helper package for financial language models",
    long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
    #url="http://example.com/mypackage",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)