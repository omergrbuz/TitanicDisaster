import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score


# importing Data
db_train = pd.read_csv("train.csv")
db_test = pd.read_csv("test.csv")
db_results = pd.read_csv("gender_submission.csv")


# Filling NaN values with median
db_test.Age.fillna(db_train.Age.median(), inplace = True)
db_train.Age.fillna(db_train.Age.median(), inplace = True)
db_test.Fare.fillna(db_test.Fare.median(), inplace = True)


# Filling NaN values with most common value
db_train.Embarked.fillna('S', inplace=True)
db_test.Embarked.fillna('S', inplace=True)


# Transforming df to array
data_train = db_train.values
data_test = db_test.values
data_results = db_results.values


# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked for X_train and X_test
X_train = np.concatenate((data_train[:,2:3], data_train[:,4:8], data_train[:,9:10], data_train[:,11:12]), axis = 1)
Y_train = np.array(data_train[:,1:2],dtype = 'int64')
X_test = np.concatenate((data_test[:,1:2],data_test[:,3:7], data_test[:,8:9], data_test[:,10:11]), axis = 1)
Y_test = data_results[:,1:2]


# Transforming categorical data with LabelEncoder and OneHotEncoder
le = LabelEncoder()
X_train[:,1] = le.fit_transform(X_train[:,1])
X_train[:,6] = le.fit_transform(X_train[:,6])
X_test[:,1] = le.fit_transform(X_test[:,1])
X_test[:,6] = le.fit_transform(X_test[:,6])

ct1 = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X_train = np.array(ct1.fit_transform(X_train))
X_test = np.array(ct1.fit_transform(X_test))

ct7 = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [7])], remainder="passthrough")
X_train = np.array(ct7.fit_transform(X_train))
X_test = np.array(ct7.fit_transform(X_test))


# Scaling Data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Building an ANN
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 12, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 12, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 10, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

ann.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, Y_train, batch_size = 28, epochs = 50)


# Prediction
pred = ann.predict(X_test)
pred = (pred > 0.5)

cm = confusion_matrix(Y_test, pred)
score = accuracy_score(Y_test, pred)
print(score)



