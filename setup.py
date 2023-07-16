from setuptools import find_packages, setup

exec(open("RACER/version.py").read())

setup(
    name="pyracer",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="Unofficial Python implementation of the RACER classification algorithm.",
    author="Arian Tashakkor, Mohammad Safaiyan",
    author_email="a77physics@gmail.com",
    url="https://github.com/Adversarian/RACER",
    long_description_content_type="text/markdown",
    keywords=["machine learning", "classification", "RACER"],
    install_requires=[
        "numpy",
        "pandas",
        "optbinning",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
