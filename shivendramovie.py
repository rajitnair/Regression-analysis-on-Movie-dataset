# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:29:46 2018

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn import metrics
#Loading Dataset 
IMDB=pd.read_csv('IMDB-Movie-Data.csv')
IMDB.describe()

#Selecting needed columns from Dataframe
Features_List=["Rank", "Year", "Runtime", "Rating", "Votes", "Revenue"]
Predict_List=["Metascore"]
Features=IMDB.loc[:, Features_List].values
Predict=IMDB.loc[:, Predict_List].values

#Adding Missing Values to the Columns
 
from sklearn.preprocessing import Imputer
imputer_features=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer_predict=Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer_features=imputer_features.fit(Features)
Features=imputer_features.transform(Features)

imputer_predict=imputer_predict.fit(Predict)
Predict=imputer_predict.transform(Predict)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scale=StandardScaler()
Features_Scaled=Scale.fit_transform(Features)
Predict_Scaled=Scale.fit_transform(Predict)
 

#Applying Backward Elimination
import statsmodels.formula.api as sm

SL = 0.05
#Features_Modeled = backwardElimination(Features, SL)

#Spliting the Dataset for Training and Testing
Features_Train, Features_Test, Predict_Train, Predict_Test=train_test_split(Features_Scaled, Predict_Scaled, test_size=0.2, random_state=0)

#Random Forest Regressor
Metascore_Predictor1=RandomForestRegressor()
Metascore_Predictor1.fit(Features_Train, Predict_Train)
Metascore_Predicted1=Metascore_Predictor1.predict(Features_Test)
print(" According to Random Forest Regression")
print ("Mean Sqaure error of the Model: " + str(mean_squared_error(Predict_Test, Metascore_Predicted1)))
print ("R squared Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted1)))



#print('Accuracy Score: ', + str(metrics.accuracy_score(Predict_Test, Metascore_Predicted1)*100,'%',sep=''))
#print("Accuracy of the model:", )
#Ridge regressor
Metascore_Predictor2=Ridge()
Metascore_Predictor2.fit(Features_Train, Predict_Train)
Metascore_Predicted2=Metascore_Predictor2.predict(Features_Test)
print(" According to Ridge Regression")
print ("Mean Sqaure error of the Model:" + str(mean_squared_error(Predict_Test, Metascore_Predicted2)))
print ("R squared Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted2)))
#Linear Regression
Metascore_Predictor3=LinearRegression()
Metascore_Predictor3.fit(Features_Train, Predict_Train)
Metascore_Predicted3=Metascore_Predictor3.predict(Features_Test)
print(" According to linear Regression")
print ("Mean Sqaure error of the Model: " + str(mean_squared_error(Predict_Test, Metascore_Predicted3)))
print ("R squared Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted3)))
#Elastic Net Regreesion
Metascore_Predictor4=ElasticNet()
Metascore_Predictor4.fit(Features_Train, Predict_Train)
Metascore_Predicted4=Metascore_Predictor4.predict(Features_Test)
print(" According to Elastic Net Regression")
print ("Mean Sqaure error of the Model: " + str(mean_squared_error(Predict_Test, Metascore_Predicted4)))
print ("R squared Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted4)))

models = pd.DataFrame({'Models':['RandomForest','Ridge','Linear','ElasticNet'],
                        'Mean Square Error':[(mean_squared_error(Predict_Test, Metascore_Predicted1)),(mean_squared_error(Predict_Test, Metascore_Predicted2)),(mean_squared_error(Predict_Test, Metascore_Predicted3)), (mean_squared_error(Predict_Test, Metascore_Predicted4))],
                          'R squared score': [(r2_score(Predict_Test, Metascore_Predicted1)),(r2_score(Predict_Test, Metascore_Predicted2)), (r2_score(Predict_Test, Metascore_Predicted3)), (r2_score(Predict_Test, Metascore_Predicted4))]})

print(models)
models.plot(kind='bar')
