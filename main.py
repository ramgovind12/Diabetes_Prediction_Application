import pandas as pd
import numpy as np
df = pd.read_csv('Data/diabetes.csv')
x = df.drop('Outcome',axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train_scaled,y_train)
prediction = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,prediction)
import joblib
joblib.dump(model,'diabetes_model.pkl')
joblib.dump(scaler,'scaler.pkl')