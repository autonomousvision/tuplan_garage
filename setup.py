import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="tuplan-garage",
    version="1.0.0",
    author="CS_Tu @ University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="tuPlan Garage of the Autonomous Vision Group.",
    url="https://github.com/autonomousvision/tuplan_garage",
    python_requires=">=3.9",
    packages=["tuplan_garage"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
