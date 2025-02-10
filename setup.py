from setuptools import setup, find_packages
import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_data={
        "SimulationFramework": [
            "Codes/*.yaml",
            "Codes/Elegant/*.yaml",
            "Codes/CSRTrack/*.yaml",
            "Codes/Ocelot/*.yaml",
            "*.yaml",
        ]
    },
)
