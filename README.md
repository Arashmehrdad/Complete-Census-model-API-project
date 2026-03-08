# Complete Census Model API Project

This repository contains my completed project for the Machine Learning DevOps Engineer program. In this project, I built and deployed a machine learning API that predicts whether a person's income is `<=50K` or `>50K` using Census Bureau data. The goal was to take the model beyond local experimentation and turn it into a tested, version-controlled, and deployable service.

## Project Overview

This project focused on the full workflow of a production-style classification system:

* cleaning and preparing Census Income data
* training a machine learning model
* evaluating performance on the overall test set and on data slices
* exposing inference through a FastAPI application
* writing unit tests for both model code and API endpoints
* validating the project with sanity checks and linting
* automating testing with GitHub Actions
* deploying the application to Render with continuous deployment enabled

Rather than stopping at model training, I treated this as an end-to-end MLOps exercise. The result is a working API backed by a trained model, supported by automated testing and deployment.

## What I Built

The final project includes:

* a preprocessing pipeline using `OneHotEncoder` and `LabelBinarizer`
* a `RandomForestClassifier` for income classification
* a training script that saves the model and preprocessing artifacts
* slice-based evaluation saved to `slice_output.txt`
* a FastAPI service with:

  * `GET /` for a welcome message
  * `POST /predict` for model inference
* unit tests for model functions, data processing, and API behavior
* CI with GitHub Actions running `flake8` and `pytest`
* deployment on Render

## Model Performance

The model was evaluated using precision, recall, and F1 score.

* Precision: 0.742
* Recall: 0.638
* F1 Score: 0.686

I also generated slice-based metrics for categorical features to inspect how performance changes across subgroups.

## Tech Stack

This project uses:

* Python 3.13
* pandas
* scikit-learn
* FastAPI
* Pydantic
* Uvicorn
* pytest
* flake8
* GitHub Actions
* Render

## Running the Project Locally

Install dependencies:

```bash
pip install -r starter/requirements.txt
```

Train the model artifacts:

```bash
python -m starter.starter.train_model
```

Run the API locally:

```bash
uvicorn starter.main:app --reload
```

Run the tests:

```bash
python -m pytest starter/tests -v
```

## Project Links

* GitHub Repository: [https://github.com/Arashmehrdad/Complete-Census-model-API-project](https://github.com/Arashmehrdad/Complete-Census-model-API-project)
* Live API: [https://complete-census-model-api-project.onrender.com](https://complete-census-model-api-project.onrender.com)

## Author

Arash Mehrdad
