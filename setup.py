import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gp-derivatives-variational-inference", 
    version="0.0.1",
    author="",
    author_email="author@example.com",
    description="scalable GP variational inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": ""},
    packages=setuptools.find_packages(),#setuptools.find_packages(where="directionalvi"),
    python_requires=">=3.6",
)