from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    """
    this function will return list of requirements
    """
    requirements:List[str] =[]
    try:
        with open('requirements.txt', 'r') as file:
            #read lines from the file
            lines = file.readlines()
            #process each line
            for line in lines:
                requirement = line.strip()
                ##ignore emppty lines and e .
                if requirement and  requirement!= '-e .':
                    requirements.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirements

setup(
    name = "flight_fare_prediction",
    version="0.0.1",
    author = "pavani Kusuma",
    author_email ="pavanikusuma2020@gmail.com",
    packages = find_packages(),
    packages_dir = {"" : "src"},
    install_requires = get_requirements()
)
