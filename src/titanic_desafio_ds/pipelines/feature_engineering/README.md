# Pipeline feature_engineering

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.18.3`.

## Overview

Pipeline dedicated to data prep and validation. 

## Pipeline inputs

* predictive_maintanence - raw dataset with features and (multiple) target columns
* features (parameter) - list of values which defines which columns are to be used as features
* target (parameter) - single value which defines which column is to be used as target

## Pipeline outputs

Validation, train and test datasets
