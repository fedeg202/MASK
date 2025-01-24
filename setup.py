import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="nome_libreria",
    version="1.0.0",
    author="Tuo Nome",
    author_email="tua_email@example.com",
    description="Breve descrizione della libreria",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fedeg202/MASK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: NO LICENCE",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'numpy>=2.1.3',
        'pandas>=2.2.3',
        'setuptools>=65.5.0'
    ],
)