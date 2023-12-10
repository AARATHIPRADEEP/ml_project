import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\admin\Downloads\obesity_ml_project.csv")
print(df)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

knn=KNeighborsClassifier(n_neighbors=5)
sv=SVC()
rf=RandomForestClassifier(random_state=1)

knn=knn.fit(X_train,y_train)
sv=sv.fit(X_train,y_train)
rf=rf.fit(X_train,y_train)
print(knn.score(X_test,y_test))

import pickle

pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(sv,open('svc_model.pkl','wb'))
pickle.dump(rf,open('rf_model.pkl','wb'))



import streamlit as st
import pickle

knn_model=pickle.load(open('knn_model.pkl','rb'))
svc_model=pickle.load(open('svc_model.pkl','rb'))
rf_model=pickle.load(open('rf_model.pkl','rb'))

def classify(num):
    if num==0:
        return 'Underweight'
    elif num==1:
        return 'Normal Weight'
    elif num==2:
        return 'Overweight'
    elif num==3:
        return 'Obese'


def main():
      st.title('Welcome to BMI Calculator')
      activities=['K-Nearest Neighbour Classifier','SVM classifier','RandomForest Classifier']
      option=st.sidebar.selectbox('Which model would you like to use?',activities)
      st.subheader(option)

      weight = st.number_input("Enter your weight (in kgs)")


      status = st.radio('Select your height format: ',
                  ('cms', 'meters', 'feet'))


      if(status=='cms'):

           height = st.number_input('Centimeters')
           try:
               bmi = weight / ((height / 100) ** 2)
           except:
               st.text("Enter some value of height")

      elif (status == 'meters'):

             height = st.number_input('Meters')

             try:
                 bmi = weight / (height ** 2)
             except:
                  st.text("Enter some value of height")

      else:

             height = st.number_input('Feet')
             try:
                bmi = weight / (((height/3.28))**2)
             except:
                  st.text("Enter some value of height")
      if(st.button('Calculate BMI')):
              st.text("Your BMI Index is {}.".format(bmi))
      if(st.button('Classify')):
              if option=='K-Nearest Neighbour Classifier':
                    st.success(classify((knn_model.predict(inputs))))

              elif option=='SVM classifier':
                    st.success(classify((svc_model.predict(inputs))))
              else:
                    st.success(classify((rf_model.predict(inputs))))

