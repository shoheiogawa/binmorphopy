import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="binmorphopy",
    version="0.0.1",
    author="Shohei Ogawa",
    author_email="ogawashohei@gmail.com",
    description="Fast morphological operations for binary images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shoheiogawa/binmorphopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
