from setuptools import setup, find_packages

setup(
    name="LegalDefAgent",
    version="0.2.0",
    #description="A legal definition retrieval and generation system.",
    #author="Leonardo Zilli",
    url="https://github.com/leonardozilli/LegalDefAgent",
    packages=find_packages(include=['LegalDefAgent*']),
    include_package_data=True,
    install_requires=[
        "polars",
        "pymilvus"
    ],
    entry_points={
        "console_scripts": [
        ]
    },
)
