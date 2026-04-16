from setuptools import setup, find_packages

setup(
    name="sql-rl-env",
    version="0.1.0",
    author="Kartik Munjal",
    description=(
        "Gymnasium-compatible RL environment for SQL query generation "
        "with formal reward signal design and reward hacking analysis"
    ),
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.26.0",
        "torch>=2.1.0",
        "sqlparse>=0.4.4",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black", "ruff"],
        "vis": ["matplotlib>=3.8.0", "seaborn>=0.13.0"],
    },
)
