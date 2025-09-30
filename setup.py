import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="signchart",
    version="0.1.28",
    author="René Alexander Ask",
    author_email="rene.ask@icloud.com",
    description="Automatically generates sign charts for polynomial functions and rational functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reneaas/signchart",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "matplotlib", "sympy", "plotmath>=0.2.18"],
    python_requires=">=3.7",
)
