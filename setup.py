import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wtphm",
    version="0.1.2",
    author="Kevin Leahy",
    description="SCADA data pre-processing library for prognostics and health"
                "management and fault detection of wind turbines",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/lkev/wtphm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy", "scipy", "matplotlib", "sklearn"],
    include_package_data=True,
    python_requires='>=3.6'
)
setuptools.find_packages()
