import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("diabetespimaindian.csv")
print(df.head())

print(df.info())

df_copy =df.copy(deep = True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(df_copy.isnull().sum())

df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)
df=df_copy
df=df.reset_index()
df.dropna(inplace=True)
print(df.isnull().sum())
Y=df['Outcome']
X=df.drop(['Outcome'],axis='columns')


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=7)

s_c = StandardScaler()
x_train = s_c.fit_transform(x_train)
x_test = s_c.fit_transform(x_test)
fil_name='scaler.sav'
pickle.dump(s_c,open(fil_name,'wb'))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_prediction = rfc.predict(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_predicted=knn.predict(x_test)




# Saving model to disk
pickle.dump(knn, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

