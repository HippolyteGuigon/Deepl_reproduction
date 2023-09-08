# Deepl_reproduction
The goal of this repository is to create a replica of the Deepl translator with the French, English and Japanese languages

## Build Status

For the moment, the main architecture relying on Google Cloud Platform with databases, cloud functions and Virtual Machine is finished. Futhermore, the pipeline starting from GCP and training the model is also over.

The first version of the model traducing from French to English is over and can be used. 

The next steps are to keep improving this model as well as launching new trainings for a translation from French to Japanese. 
Also, a user-friendly interface using streamlit will be developped. Finally, a text-to-speech model will be implemented for nicer translations.

Throughout the project, if you see any improvements that could be made in the code, do not hesitate to reach out at
Hippolyte.guigon@hec.edu. I will b delighted to get some insights !

## Code style

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f deepl_reproduction.yml```

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot

![alt text](https://raw.githubusercontent.com/HippolyteGuigon/Deepl_reproduction/main/ressources/logo.webp)


## How to use ?
