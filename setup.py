from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="nlp-distil-bert-tree",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="BERT to Decision Tree knowledge distillation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/nlp-distil-bert-tree",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
) 