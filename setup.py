import setuptools

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sourcesep",
    description="Source separation for multiplexed modulator imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["sourcesep"],
    python_requires='>=3.7',
)
