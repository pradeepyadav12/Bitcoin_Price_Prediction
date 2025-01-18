# Bitcoin Price Prediction using Machine Learning in Python

![Bitcoin Image](https://github.com/pradeepyadav12/Bitcoin_Price_Prediction/blob/main/bitcoin.webp)

# Overview
The article outlines a step-by-step process to build a predictive model for Bitcoin prices. It emphasizes the importance of using historical price data and technical indicators to train the model effectively. The approach involves data collection, preprocessing, feature engineering, model selection, training, and evaluation.

# Objectives
- Data Collection and Preprocessing: Gather historical Bitcoin price data and preprocess it to handle missing values, normalize features, and create relevant technical indicators.

- Feature Engineering: Develop features such as Exponential Moving Average (EMA) and Moving Average Convergence Divergence (MACD) to capture essential market trends.

- Model Development: Utilize machine learning algorithms, specifically the XGBoost regressor, to build a model that can predict future Bitcoin prices based on the engineered features.

- Model Evaluation: Assess the model's performance using appropriate metrics to ensure its accuracy and reliability in predicting Bitcoin prices.


# Dataset
The data for this project is sourced from the kaggle dataset:
- **Dataset Link:** [Bitcoin Dataset](https://www.kaggle.com/datasets/varpit94/bitcoin-data-updated-till-26jun2021?resource=download)

## Code

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```













