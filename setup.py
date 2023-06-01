from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["deepdiff>=5", "h5py>=2.10", "munch>=2.5", "numpy>=1.19", "progressbar2>=3",
                "PyQt5>=5.1", "PyYAML>=5.3", "mpl-axes-aligner>=1.1", "lox>=0.11"]

setup(
    name="AcceleratorSimFrame",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="James Jones",
    author_email="james.jones@stfc.ac.uk",
    description="A python framework for particle accelerator simulations",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/VELA-CLARA-software/SimFrame",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
	package_data={'SimulationFramework': ['Codes/*.yaml', 'Codes/Elegant/*.yaml', 'Codes/CSRTrack/*.yaml','*.yaml']},
)
