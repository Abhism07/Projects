import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetespimaindian.csv')
real_x = data.iloc[:, 0:8].values

real_y = data.iloc[:, 8].values
training_x, testing_x, training_y, testing_y = train_test_split(real_x,real_y,test_size=0.25,random_state=0)
#print("training X =",training_x)
s_c = StandardScaler()
training_x = s_c.fit_transform(training_x)

#print("training X =",training_x)
test_x = s_c.transform(testing_x)
fil_name='scaler.sav'
pickle.dump(s_c,open(fil_name,'wb'))
#print("testing x",test_x)
cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
cls.fit(training_x,training_y)
pickle.dump(cls, open('db.pkl','wb'))

model = pickle.load(open('db.pkl','rb'))
print(model.predict([[1, 89, 66, 23, 94, 28.1, 0.167, 21]]))
y_pred = cls.predict(test_x)
c_m = confusion_matrix(testing_y, y_pred)
acc = accuracy_score(testing_y, y_pred)
print(data.isnull().values.any())
print(c_m)
print(acc)

