# Default Detector

## Overview
In this project, I built an insurance risk detection model that detects 96% of prospective defaulters using a CatBoost Classifier. I demonstrate my ability to produce high-quality data science deliverables in the field of insurance. 

## Steps

The repository has a detailed description of my decision-making steps. A basic overview of the steps taken is the following:

1. EDA and an investigation into the features
2. Separating features by Imputing strategy
3. Preprocessing using SKLearn pipelines (Encoding, Scaling, Imputing with strategies tuned depending on the feature using custom transformers)
4. Feature Engineering
5. Testing base versions of different models and settling on CatBoost.
6. Sequential Feature Selection to find the best features
7. Hyperparameter tuning a CatBoost model with my selected features
8. Model Evaluation
9. A feature importance analysis using SHAP values and feature permutation

## Notes

- There are many decision points in this project that allow me to make competing assumptions. Broadly, I prioritise minimising the chance of a defaulter going undetected (Hence my choice of recall as a metric). However, I do acknowledge that this comes at the cost of revenue to the loaner.
- This type of model is not used to make predictions, only to inform the investigation of prospective loan recipients. Some of the metrics are suboptimal, but this is to be expected when working with such an imbalanced dataset.
- I had some level of uncertainty about SequentialFeatureSelection. I considered not using it in favour of built-in regularisation, but decided to go with it because regularisation and SequentialFeatureSelection take place at different layers in the design process. While regularisation is good, SequentialFeatureSelection can give what are objectively the best features. Unfortunately, my machine had several resource-related errors (RAM) so I did end up having to skip it.
- I implemented ElasticNet for regularisation, but decided to remove it. I knew from the start that I was going to be using tree-based models, which do not benefit from feature scaling. I still used scaling for compatibility with ElasticNet, but I decided to use built-in alternatives. I chose built-in regularisation because:
  1. It's a much more computationally efficient solution
  2. It's a much easier process to follow
  3. I generally prefer to go with built-in solutions where possible. For example in this case, while ElasticNet CAN be used for regularisation in classification tasks, that is not what it is designed for. I believe it is safer to use a built-in solution in cases like this.
- I would have liked to spend more time iterating over my solution. For example, with feature engineering, I would have liked to go back and change/engineer new features after my initial regularisation-based feature importance.
