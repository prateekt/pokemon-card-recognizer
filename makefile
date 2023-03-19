conda_dev:
	conda env remove -n card_rec_env
	conda env create -f conda.yaml

conda_deploy:
	conda env remove -n card_rec_env
	conda env create -f conda_deploy.yaml

build_reference:
	rm -rf card_recognizer/reference/data
	python card_recognizer/reference/core/build.py $(PGTCGSDK_API_KEY)

eval_master_on_reference:
	python card_recognizer/reference/eval/eval_master_on_reference.py

eval_pipeline_on_reference:
	python card_recognizer/reference/eval/eval_pipeline_on_reference.py

build:
	rm -rf dist
	rm -rf build
	rm -rf pokemon_card_recognizer.egg*
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*

clean:
	rm -rf dist
	rm -rf build
	rm -rf pokemon_card_recognizer.egg*
	rm -rf .pytest_cache

