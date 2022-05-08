from google.colab import files
data_to_upload = files.upload()
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

df= pd.read_csv("Admission_Predict.csv")
TOEFL  = df["TOEFL Score"].tolist()
results = df["Chance of admit"].tolist()
grescore = df["GRE Score"].tolist()

colors = []
for data in results:
  if data==1:
    colors.append("green")
  else:
    colors.append("red")

factors = df[["TOEFL Score" , "GRE Score"]]
results = df["Chance of admit"]
from sklearn.model_selection import train_test_split
score_train , score_test , results_train , results_test = train_test_split(factors, results, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler 
scx = StandardScaler()
score_train = scx.fit_transform(score_train)
score_test  = scx.transform(score_test)
print(score_train[0:10])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train,results_train)
results_prediction = classifier.predict(score_test)
from sklearn.metrics import accuracy_score
print("the accuracy of the predication model is :" , accuracy_score(results_test, results_prediction))

user_grescore = int(input("Enter GRE Score of the student -> "))
user_scores = int(input("Enter the TOEFL Score of the student -> "))

user_test = scx.transform([[user_grescore, user_scores]])

user_results_pred = classifier.predict(user_test)
if user_results_pred[0] == 1:
      print("This student lives")
else:
      print("nahh he dead fam") 
