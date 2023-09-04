conda_dev:
	conda env remove -n card_rec_env
	conda env create -f conda.yaml

build_reference:
	rm -rf card_recognizer/reference/data
	python card_recognizer/reference/core/build.py $(PGTCGSDK_API_KEY)

eval_master_on_reference:
	python card_recognizer/reference/eval/eval_master_on_reference.py

eval_pipeline_on_reference:
	python card_recognizer/reference/eval/eval_pipeline_on_reference.py

build:
	rm -rf dist
	hatch build

publish:
	hatch publish

clean:
	rm -rf dist
	rm -rf .pytest_cache

