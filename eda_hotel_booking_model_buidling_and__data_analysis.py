# -*- coding: utf-8 -*-
"""eda-hotel-booking-Model_Buidling_and _data-analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MAmxRX7_0C1YcHMzDOabejT6JC6RUjvD

# Hotel Booking Data

# Importing Liabraries
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""# Loading Dataset

"""

data= pd.read_csv("/content/hotel_bookings 2.csv")
data

"""# EDA And Data Cleaning"""

data.head()

data.shape

data.info()

data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

data.info()

data.describe(include='object')

#check the unique values
for col in data.describe(include='object').columns:
    print(col)
    print(data[col].unique())
    print('-'*50)



#checking the null values
data.isnull().sum()

data.isnull().sum()

data.describe()

data = data[data['adr']<5000]

data.describe()



data.head().T

"""# Model Building"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

data = data.drop(columns=['country'])

#check the unique values
for col in data.describe(include='object').columns:
    print(col)
    print(data[col].unique())
    print('-'*50)

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder for each categorical feature
le_hotel = LabelEncoder()
le_arrival_date_month = LabelEncoder()
le_meal = LabelEncoder()
le_market_segment = LabelEncoder()
le_distribution_channel = LabelEncoder()
le_reserved_room_type = LabelEncoder()
le_assigned_room_type = LabelEncoder()
le_deposit_type = LabelEncoder()
le_customer_type = LabelEncoder()
le_reservation_status = LabelEncoder()

# Fit and transform each feature
data['hotel'] = le_hotel.fit_transform(data['hotel'])
data['arrival_date_month'] = le_arrival_date_month.fit_transform(data['arrival_date_month'])
data['meal'] = le_meal.fit_transform(data['meal'])
data['market_segment'] = le_market_segment.fit_transform(data['market_segment'])
data['distribution_channel'] = le_distribution_channel.fit_transform(data['distribution_channel'])
data['reserved_room_type'] = le_reserved_room_type.fit_transform(data['reserved_room_type'])
data['assigned_room_type'] = le_assigned_room_type.fit_transform(data['assigned_room_type'])
data['deposit_type'] = le_deposit_type.fit_transform(data['deposit_type'])
data['customer_type'] = le_customer_type.fit_transform(data['customer_type'])
data['reservation_status'] = le_reservation_status.fit_transform(data['reservation_status'])

"""# Convert categorical variables to numerical using Label Encoding
label_encoders = {}
categorical_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le"""

data.head()

missing_values = data.isnull().sum()
print(missing_values)

data.drop(['agent', 'company','children'], axis=1, inplace=True)

data.drop(['reservation_status_date'], axis=1, inplace=True)

missing_values = data.isnull().sum()
print(missing_values)

x = data.drop("is_canceled",axis=1)
y = data['is_canceled']
#split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

lr_clf = LogisticRegression()
lr_clf.fit(x_train,y_train)

y_pred = lr_clf.predict(x_test)
y_pred[20:25] # Y predicted

y_test[20:25] # y actual

cnf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cnf_matrix)

clf_report = classification_report(y_test,y_pred)
print("classification_report is :\n",clf_report)

y_pred_prob = lr_clf.predict_proba(x_test)
y_pred_prob

fpr,tpr,thresh = roc_curve(y_test,y_pred_prob[:,1])

thresh

plt.title("ROC Curve")
plt.plot(fpr,tpr)
plt.xlabel("Flase Positive Rate")
plt.ylabel("True Positive Rate")

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)



"""# Data Analysis and Visualization"""

data['is_canceled'].value_counts()
cancelled_perce=data['is_canceled'].value_counts(normalize=True)
print(cancelled_perce)

plt.figure(figsize = (5,4))
plt.title('Reservation status count')
plt.bar(['Not canceled','Canceled'],data['is_canceled'].value_counts(),edgecolor='k',width=0.7)
plt.show()

"""here is clear that 37% people canceled the booking that is high percentage"""

plt.figure(figsize = (8,4))
ax = sns.countplot(x = 'hotel', hue = 'is_canceled',data=data, palette= 'Blues')
legend_labels,_ = ax. get_legend_handles_labels()
plt.title('Reservation status in different hotels', size=20)
plt.xlabel('hotel')
plt.ylabel('number of reservations')
plt.legend(['not_canceled','canceled'])
plt.show()

resort_hotel = data[data['hotel'] == 'Resort Hotel']
resort_hotel['is_canceled'].value_counts(normalize = True)

city_hotel = data[data['hotel'] == 'City Hotel']
city_hotel['is_canceled'].value_counts(normalize = True)

resort_hotel = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel = city_hotel.groupby('reservation_status_date')[['adr']].mean()

plt.figure(figsize = (20,8))
plt.title('Average Daily Rate in City and Resort Hotel', fontsize=30)
plt.plot(resort_hotel.index, resort_hotel['adr'], label = 'Resort Hotel')
plt.plot(city_hotel.index, city_hotel['adr'], label = 'City Hotel')
plt.legend(fontsize = 20)
plt.show()

data['month'] = data['reservation_status_date'].dt.month
plt.figure(figsize = (16,8))
ax = sns.countplot(x='month',hue='is_canceled',data=data,palette='bright')
legend_labels,_ = ax.get_legend_handles_labels()
ax.legend(bbox_to_anchor = (1,1))
plt.title('Reservation status per month',size=20)
plt.xlabel('month')
plt.ylabel('number of reservations')
plt.legend(['not canceled','canceled'])
plt.show()

cancelled_data = data[data['is_canceled'] == 1]
top_10_country = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(10,10))
plt.title('Top 10 countries with reservation canceled')
plt.pie(top_10_country, autopct='%.2f',labels = top_10_country.index)
plt.show()

data['market_segment'].value_counts()

data['market_segment'].value_counts(normalize=True)

cancelled_data['market_segment'].value_counts(normalize=True)

cancelled_data_adr = cancelled_data.groupby('reservation_status_date')[['adr']].mean()
cancelled_data_adr.reset_index(inplace=True)
cancelled_data_adr.sort_values('reservation_status_date',inplace=True)

not_cancelled_data = data[data['is_canceled'] == 0]
not_cancelled_data_adr = not_cancelled_data.groupby('reservation_status_date')[['adr']].mean()
not_cancelled_data_adr.reset_index(inplace=True)
not_cancelled_data_adr.sort_values('reservation_status_date',inplace=True)

plt.figure(figsize = (20,6))
plt.title('Average Daily Rate')
plt.plot(not_cancelled_data_adr['reservation_status_date'],not_cancelled_data_adr['adr'],label='not cancelled')
plt.plot(cancelled_data_adr['reservation_status_date'],cancelled_data_adr['adr'],label='cancelled')
plt.legend()

cancelled_data_adr = cancelled_data_adr[(cancelled_data_adr['reservation_status_date']>'2016') & (cancelled_data_adr['reservation_status_date']<'2017')]
not_cancelled_data_adr = not_cancelled_data_adr[(not_cancelled_data_adr['reservation_status_date']>'2016') & (not_cancelled_data_adr['reservation_status_date']<'2017')]

plt.figure(figsize = (20,6))
plt.title('Average Daily Rate',fontsize=30)
plt.plot(not_cancelled_data_adr['reservation_status_date'],not_cancelled_data_adr['adr'],label='not cancelled')
plt.plot(cancelled_data_adr['reservation_status_date'],cancelled_data_adr['adr'],label='cancelled')
plt.legend(fontsize=20)