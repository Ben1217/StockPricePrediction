from setuptools import setup, find_packages

setup(
    name="stock-prediction-dashboard",
    version="1.0.0",
    author="Tan Yee Hern",
    author_email="your.email@student.utar.edu.my",
    description="Stock Price Prediction and Portfolio Optimization Dashboard",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-prediction-dashboard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
