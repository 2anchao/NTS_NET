import torch
from setuptools import setup
from setuptools import find_packages



setup(
    name="NTS_NET",
    version="1.0",
    author = "anchao",
    author_email = "anchao@supremind.com",
    url="https://github.com/yangze0930/NTS-Net",
    description="NTS_NET in pytorch",
    packages=find_packages(exclude=(
        "core",
        "datasets"
        "tools",
    )),
)
