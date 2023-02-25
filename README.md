# titanic_desafio_ds

## Dataset
* Data can be obtained from the link: [https://www.kaggle.com/competitions/titanic/data]
* But the raw data is also here in the repository

## How to run

* Python Version:
    `python -V`:
        `Python 3.8.11`

* Virtual env
    `python -m venv venv`
    `source venv/bin/activate`

* Install packages:
    `pip install -r src/requirements.txt`

* To run the project:
    `kedro run`

* To access notebooks:
    `kedro jupyter lab`

* To access API (fast-api) with model trained (first `kedro run`):
    `kedro fast-api run`

* To access metrics from training:
    `kedro mlflow ui`
