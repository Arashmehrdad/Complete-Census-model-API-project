# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a RandomForestClassifier from scikit-learn to predict whether a person's annual income is `<=50K` or `>50K` based on Census Income data. The preprocessing pipeline uses a OneHotEncoder for categorical variables and a LabelBinarizer for the target label. The model, encoder, and label binarizer are saved as pickle files for later inference through a FastAPI application.

## Intended Use

The model is intended for educational purposes as part of a machine learning engineering project. Its purpose is to demonstrate data preprocessing, model training, slice-based evaluation, API deployment, and automated testing. It should not be used for real hiring, lending, insurance, legal, or other high-stakes decision-making because the dataset reflects historical social patterns and may contain bias.

## Training Data

The training data comes from the Census Income dataset provided in the project starter files. The dataset contains demographic and employment-related features such as age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, and native-country. The target variable is salary, grouped into two classes: `<=50K` and `>50K`.

The data was split into training and test sets using an 80/20 train-test split. Categorical features were encoded using one-hot encoding before model training.

## Evaluation Data

The evaluation data is the held-out 20 percent test split from the same Census Income dataset. The same preprocessing pipeline used for training was applied to the evaluation data, using the fitted encoder and label binarizer from the training stage.

In addition to overall evaluation, slice-based metrics were generated for categorical features and written to `starter/model/slice_output.txt`.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Overall model performance on the test set:
- Precision: 0.742
- Recall: 0.638
- F1 Score: 0.686

These metrics indicate that the model performs reasonably well for this educational task, although performance varies across different categorical slices.

## Ethical Considerations

This model uses demographic and socioeconomic features, including race, sex, marital-status, and native-country. These features may introduce or reinforce historical biases present in the dataset. Even if the model performs acceptably in aggregate, it may behave differently across population subgroups.

Because of this, the model should not be used in any real-world setting where decisions affect people's opportunities, rights, or access to services. Slice-based evaluation was included in this project to help inspect whether performance differs across categories, but this does not remove the risk of unfairness.

## Caveats and Recommendations

This model was built for coursework and demonstration purposes, not for production use. The dataset is relatively old and may not reflect current labor market conditions. The model also depends on the quality of preprocessing and the representativeness of the available data.

I recommend the following improvements for future work:
- compare multiple model types instead of using a single baseline model
- perform hyperparameter tuning
- add deeper fairness analysis across demographic groups
- improve monitoring and validation for deployment
- avoid using this model for any real high-stakes application