from setuptools import setup, find_packages

setup(
    name="cloudybot",
    version="0.1.0",
    author="Akshay Mittal",
    author_email="akshaymittal143@gmail.com",  # You may want to verify this email
    description="An AI-powered DevOps assistant chatbot",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/akshaymittal143/CloudyBot-DevOps-AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
