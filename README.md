# eda-hotel-booking-Model_Buidling_and-_data-analysis

Hotel Booking Cancellation Prediction

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction

This project aims to predict hotel booking cancellations using a dataset of hotel reservations. We will perform exploratory data analysis, feature engineering, and develop a predictive model to help hotels understand and reduce booking cancellations.

## Importing Libraries

First, we import the necessary Python libraries for data analysis and visualization:
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

Importing Dataset

We load the hotel booking dataset:

 

data = pd.read_csv("/kaggle/input/hotel-booking/hotel_booking.csv")

The dataset contains information about hotel bookings, including guest details, booking channels, and more.
Exploratory Data Analysis (EDA)

We start with exploratory data analysis to gain insights into the dataset:

 

# Display the first few rows of the dataset
data.head()

# Check the shape and data types of the dataset
data.shape
data.info()

# Convert reservation_status_date to a datetime data type
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

# Summary statistics of the dataset
data.describe(include='object')

# Check unique values in categorical columns
for col in data.describe(include='object').columns:
    print(col)
    print(data[col].unique())
    print('-' * 50)

Feature Engineering

After the initial data exploration, we proceed to feature engineering to prepare the data for modeling. This includes handling missing values and outliers:

 

# Check for missing values
data.isnull().sum()

# Drop columns with many missing values and irrelevant columns
data.drop(['agent', 'company', 'name', 'email', 'phone-number', 'credit_card'], axis=1, inplace=True)
data.dropna(inplace=True)

# Handle outliers, e.g., remove entries with extreme values in 'adr'
data = data[data['adr'] < 5000]

Model Development and Evaluation

We will develop predictive models to determine the probability of booking cancellations and evaluate their performance:
Explore the data distribution and visualize the impact of different factors on booking cancellations
Develop predictive models to determine the probability of booking cancellations
Evaluate the model's performance using relevant metrics
Compare different models and select the best one

# Fine-tune the selected model and make predictions using LogisticRegression
Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Load the dataset
data = pd.read_csv("hotel_booking.csv")

Data preprocessing
# Drop unnecessary columns and rows with missing values
data.drop(['agent', 'company', 'name', 'email', 'phone-number', 'credit_card'], axis=1, inplace=True)
data.dropna(inplace=True)

Convert categorical variables to numerical using Label Encoding
label_encoders = {}
categorical_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
 Split the data into features and target variable
X = data.drop('is_canceled', axis=1)
y = data['is_canceled']

Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Train a Logistic Regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

Make predictions on the test set
y_pred = clf.predict(X_test)

Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

# Fine-tune the selected model and make predictions using RandomForestClassifier

 Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
  Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

  Make predictions on the test set
y_pred = clf.predict(x_test)

  Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

Accuracy: 1.00
Confusion Matrix:
 [[22401     0]
 [    1 13415]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22401
           1       1.00      1.00      1.00     13416

    accuracy                           1.00     35817
   macro avg       1.00      1.00      1.00     35817
weighted avg       1.00      1.00      1.00     35817

 Conclusion

In this project, we have analyzed a hotel booking dataset and developed predictive models to understand and predict booking cancellations. The insights gained from this analysis can help hotels reduce cancellations and improve customer satisfaction.
