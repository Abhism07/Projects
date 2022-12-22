from flask import Flask,render_template,request
import os
import pickle
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
app = Flask(__name__)
model = pickle.load(open('db.pkl', 'rb'))

@app.route('/',methods=['GET','POST'])
def hello():
     return render_template('index.html')

@app.route('/home',methods=['GET','POST'])
def home():

      return render_template('tpp.html')
   
  
if __name__ == "__main__":
    app.run(debug=True)