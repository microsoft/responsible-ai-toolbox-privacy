
.PHONY: all-tests
all-tests: lint unit-tests

.PHONY: unit-tests
unit-tests:
	@echo "Running unit tests"
	@pytest tests

.PHONY: lint
lint:
	@echo "Running lint"
	@flake8 . --max-line-length=128 --count --select=E9,F63,F7,F82,E501,E302,W605,W293,E225,F841,E117,E127,E713,E111,E303,E101,W292 --show-source --statistics
	@flake8 . --max-line-length=128 --count --exit-zero --max-complexity=10 --statistics
