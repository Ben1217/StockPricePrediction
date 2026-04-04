from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="quantvision",
    version="2.0.0",
    description="FastAPI and React workspace for stock analysis, forecasting, and portfolio research",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
