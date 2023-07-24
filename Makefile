activate_environment:
	mamba init
	mamba activate deepl_reproduction_env

setup:
	python3 setup.py install

connect_gcp:
	gcloud auth login

load_google_creditentials:
	export GOOGLE_APPLICATION_CREDENTIALS=deepl_api_key.json
