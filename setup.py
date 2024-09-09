from setuptools import setup,find_packages
from typing import List

project_name="Dogs_Vs_Cats Classification"

HYPEN_e_DOT="-e ."

def get_packages(file_path :str) -> List[str]:
    #this function is used to iterate the packages names written in the requirements.txt file
    Requirements=[]
    with open(file_path,'r') as file:
        Requirements=file.readlines()

    #now replace the \n with ""(white space)
    Requirements=[escape_sequence.replace("\n","") for escape_sequence in Requirements]

    #now remove the -e . in the Requirements list
    if HYPEN_e_DOT in Requirements:
        Requirements.remove(HYPEN_e_DOT)

    #now return the Requirements list
    return Requirements




setup(
    name=project_name,
    version='0.0.1',
    description="project is about to classify image animal is a Dog or Cat",
    author="Ankit_Gaur",
    author_email="ankitparashar000@gmail.com",
    packages=find_packages(),
    install_requires=get_packages("requirements.txt")
)