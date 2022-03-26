conda_dev:
	conda env remove -n card_rec_py39
	conda env create -f conda_osx.yaml

conda_deploy:
	conda env remove -n card_rec_py39
	conda env create -f conda_linx.yaml

build_reference:
	rm -rf card_recognizer/reference/data
	python card_recognizer/reference/core/build.py

eval_master_on_reference:
	python card_recognizer/reference/eval/eval_master_on_reference.py

eval_pipeline_on_reference:
	python card_recognizer/reference/eval/eval_pipeline_on_reference.py
