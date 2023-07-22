activate_environment:
	mamba init
	mamba activate deepl_reproduction_environment

setup:
	python3 setup.py install