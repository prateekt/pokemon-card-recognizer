build_reference:
	rm -rf card_recognizer/reference/data
	python card_recognizer/reference/core/build.py

eval_master_on_reference:
	python card_recognizer/reference/eval_scripts/eval_master_on_reference.py

eval_pipeline_on_reference:
	python card_recognizer/reference/eval_scripts/eval_pipeline_on_reference.py
