.PHONY: install run test eval gate compare docker lint

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload

test:
	pytest -q

eval:
	python -m evals.run_eval

gate:
	python -m evals.run_eval --gate

compare:
	python -m evals.run_eval --compare-modes

lint:
	ruff check app evals tests

docker:
	docker build -t secondbrain-rag . && docker run --rm -p 8000:8000 secondbrain-rag
