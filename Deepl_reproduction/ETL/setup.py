from setuptools import setup, find_packages

setup(
    name="Deepl_reproduction",
    version="0.1.0",
    packages=find_packages(
        include=["workspace/Deepl_reproduction", "workspace/Deepl_reproduction.*"]
    ),
    description="Python programm for creating a replica\
        of the Deepl traduction application",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Deepl_reproduction",
)