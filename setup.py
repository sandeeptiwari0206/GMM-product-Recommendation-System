from setuptools import setup, find_packages

setup(
    name="gmm-recommendation",
    version="1.0.0",
    description="GMM-based B2B Product Recommendation System",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.2.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "joblib>=1.2.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "aws": ["boto3>=1.26.0", "sagemaker>=2.140.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
)
