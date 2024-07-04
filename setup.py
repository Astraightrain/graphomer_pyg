from setuptools import setup, find_packages

setup(
    name="graphomer",
    version="0.1.0",
    author="Astraightrains",
    author_email="sanice1229@gmail.com",
    description="graphomer implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Astraightrains/graphomer",
    packages=find_packages(),
    install_requires=["torch", "torch-geometric"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
