MYSQL_ROOT_PASSWORD:=Bisma4And7 

activate_environment:
	mamba activate deepl_reproduction_env

setup:
	python3 setup.py install

connect_gcp:
	gcloud auth login

load_google_creditentials:
	export GOOGLE_APPLICATION_CREDENTIALS=deepl_api_key.json

launch_front_database_docker_image:
	docker build -t front_database_image:latest  -f Dockerfile-front-database .
	docker run -d -p 3306:3306 --name local-mysql-container front_database_image

launch_streamlit:
	streamlit run app.py --server.fileWatcherType none