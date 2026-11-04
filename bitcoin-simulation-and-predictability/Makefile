.PHONY: install test run quick

install:
	pip install -e ".[dev]"

test:
	pytest -v

run:            ## full study: tables in results/, figures in figures/ (~8 min)
	python -m btcsim

quick:          ## everything except the walk-forward models
	python -m btcsim --skip-prediction
