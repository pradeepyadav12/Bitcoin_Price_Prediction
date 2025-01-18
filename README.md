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

## Importing Dataset
```
df = pd.read_csv('bitcoin.csv')
df.head()
```

## data Shape
```
df.shape
```

## Describe data
```
df.describe()
```

## Data Information
```
df.info()
```

## Exploratory Data Analysis

```
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
```

## Closing Price
```
df[df['Close'] == df['Adj Close']].shape, df.shape
```

## Adj Close
```
df = df.drop(['Adj Close'], axis=1)
```

## Null values
```
df.isnull().sum()
```

## Distribution plot of the OHLC data
```
features = ['Open', 'High', 'Low', 'Close']

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.distplot(df[col])
plt.show()
```

## Boxplot of the OHLC data
```
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.boxplot(df[col])
plt.show()
```

## Feature Engineering
```
splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date']) 

df.head()

# This code is modified by Susobhan Akhuli
```

## Barplot of the mean price of the bitcoin year wise

```
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()
```

## First five rows of the data
```
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()
```

```df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
```

## Pie chart for data distribution across two labels
```
plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()
```

## Heatmap to find the highly correlated features

```
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
```

##  Build our model
```
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)
#We do not use train test split, rather use the first 70% data to train and last 30% to test
X_train, X_valid, Y_train, Y_valid = X_train, X_valid, Y_train, Y_valid = features[:len(features)//7],features[len(features)//7:],target[:len(features)//7],target[len(features)//7:]
```

## Model Development and Evaluation: like (Logistic Regression, Support Vector Machine, XGBClassifier)


```
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
```

## Confusion matrix for the validation data

```
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()

# This code is modified by Susobhan Akhuli
```

## Conclusion:
- We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.









