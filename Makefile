MYSQL_ROOT_PASSWORD:=Bisma4And7 

activate_environment:
	mamba init
	mamba activate deepl_reproduction_env

setup:
	python3 setup.py install

connect_gcp:
	gcloud auth login

load_google_creditentials:
	export GOOGLE_APPLICATION_CREDENTIALS=deepl_api_key.json

launch_front_database_docker_image:
	docker build -t local-mysql .
	docker run -d -p 3306:3306 --name local-mysql-container local-mysql
