from setuptools import find_packages, setup
from typing import List

HYPEN_DOT = '-e .'
def get_requirements(requirements_path:str)->List[str]:
    '''
    Functions returns the list of requirements
    '''
    requirements = []
    with open(requirements_path,'r') as obj:
        requirements = obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
    if HYPEN_DOT in requirements:
        requirements.remove(HYPEN_DOT)
        
    return requirements

setup(
    name = 'ChurnPredictionProject',
    version='0.0.1',
    author='Harrykesh',
    author_email='harikeshv630@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
)