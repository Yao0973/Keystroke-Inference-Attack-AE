"""Setup script for the artifact-evaluation helper package."""

from pathlib import Path
from typing import List

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements() -> List[str]:
    requirements: List[str] = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="keystroke-inference-artifact",
    version="0.1.0",
    author="Tao Gu, Yao Wang",
    author_email="tao.gu@mq.edu.au",
    description="Artifact-evaluation utilities for keystroke/PIN inference research",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
