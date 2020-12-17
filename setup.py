from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["deepdiff>=5", "h5py>=3.1", "munch>=2.5", "numpy>=1.19", "progressbar2>=3",
                "PyQt5>=5.1", "PyYAML>=5.3", "mpl-axes-aligner>=1.1",
                "MasterLattice>=0.0.1"]

setup(
    name="AcceleratorSimFrame",
    version="0.0.1",
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
	data_files=[
	('../../SimulationFramework/Codes', ['SimulationFramework/Codes/elementkeywords.yaml', 'SimulationFramework/Codes/type_conversion_rules.yaml']),
	('../../SimulationFramework/Codes/CSRTrack', ['SimulationFramework/Codes/CSRTrack/csrtrack_defaults.yaml']),
	('../../SimulationFramework/Codes/Elegant', ['SimulationFramework/Codes/Elegant/commands_Elegant.yaml',
	'SimulationFramework/Codes/Elegant/elements_Elegant.yaml', 'SimulationFramework/Codes/Elegant/keyword_conversion_rules_elegant.yaml']),
	]
)
