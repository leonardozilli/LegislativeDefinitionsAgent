from setuptools import setup, find_packages

setup(
    name="LegalDefAgent",  # Name of your package
    version="0.1.0",       # Initial version
    #description="A legal definition retrieval and generation system.",
    #author="Your Name",
    #author_email="your.email@example.com",
    #url="https://github.com/yourusername/LegalDefAgent",  # Replace with your repository URL
    packages=find_packages(),  # Automatically find packages in your directory structure
    include_package_data=True, # Include files listed in MANIFEST.in
    install_requires=[
        "polars",
        "pymilvus"
    ],
    entry_points={
        "console_scripts": [
            "legal-def-agent=app.app:main",  # Example CLI entry point
            "build-defdb=LegalDefAgent.src.db.builder:build_database"

        ]
    },
)
