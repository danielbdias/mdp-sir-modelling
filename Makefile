install.dependencies:
	pip install -r requirements.txt

run.tests:
	nose2 --start-dir ./tests --project-directory ./sir_modelling --coverage ./sir_modelling

run.tests.coverage:
	nose2 --start-dir ./tests --project-directory ./sir_modelling --coverage ./sir_modelling --coverage-report html --coverage-report term --with-coverage

run.tests.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.tests' \
		.

run.lint:
	pylint --rcfile=.pylint.cfg sir_modelling

run.lint.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.lint' \
		.

run.notebooks:
	jupyter lab
