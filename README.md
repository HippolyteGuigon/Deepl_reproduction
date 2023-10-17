# Deepl_reproduction
The goal of this repository is to create a replica of the Deepl translator with the French, English and Japanese languages

## Build Status

For the moment, the main architecture relying on Google Cloud Platform with databases, cloud functions and Virtual Machine is finished. Futhermore, the pipeline starting from GCP and training the model is also over.

The first version of the model traducing from French to English anf French to Japanese is over and can be used with a streamlit application.

The next steps are to implement a text-to-speech model for nicer translations and keep improving the translation model.

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

There are three ways to run this application:

* The application has been deployed on Cloud Run and can be launched directly via this link: https://deepl-app-7mas2kox4q-uc.a.run.app

* Classical way: After having dealt with the installation step, just run ```streamlit run app.py``` then select the language you want to translate to and enjoy the result !

* Run with Docker:
    * Make sure you have Docker installed, if you don't, just follow the steps listed here: ```https://docs.docker.com/engine/install/ubuntu/```
    * Build the image with the following command: ```docker build -t deepl_app:latest -f Dockerfile-model-loading .``` or with the MakeFile ```make build_deepl_image```
    * Run the Docker Image with ```docker run -p 8501:8501 deepl_app:latest``` or with the MakeFile ```make launch_streamlit```
    * Copy paste this link in your browser: ```http://0.0.0.0:8501```
    * You're good to translate !