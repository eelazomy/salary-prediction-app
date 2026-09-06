import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st
#======================
df=pd.read_csv(r"C:\Users\smc\Desktop\projecs\Salary prediction system\Salary Data.csv")
#print(df.head())
#=====================
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Years of Experience"]=df["Years of Experience"].fillna(df["Years of Experience"].mean())
df["Salary"]=df["Salary"].fillna(df["Salary"].mean())
#=====================
x=df.drop("Salary",axis=1)
y=df["Salary"]
#=======================
le_gender=LabelEncoder()
le_job=LabelEncoder()
le_education=LabelEncoder()
x["Job Title"]=le_job.fit_transform(x["Job Title"])
x["Education Level"]=le_education.fit_transform(x["Education Level"])
x["Gender"]=le_gender.fit_transform(x["Gender"])
#=======================
#print(x.info())
#=======================
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#======================
log=LinearRegression().fit(x_train,y_train)
#===========================================
Age=st.number_input("Age")
Gender=st.selectbox("Gender",["Male","Female"])
Education_Level=st.selectbox("Education_Level",["PhD","Master's","Bachelor's"])
Job_Title=st.selectbox("Job Title",['Software Engineer','Data Analyst','Senior Manager','Sales Associate','Director','Marketing Analyst','Product Manager','Sales Manager','Marketing Coordinator','Senior Scientist','Senior Software Architect','Junior Research Scientist','Senior Financial Manager','Senior HR Specialist','Senior Data Engineer','Junior Operations Coordinator','Director of HR','Senior Operations Coordinator','Junior Financial Advisor','Director of Engineering'])
Years_of_Experience=st.number_input("Years of Experience")
if st.button("predict"):
    Gender=le_gender.transform([Gender])[0]
    Education_Level=le_education.transform([Education_Level])[0]
    Job_Title=le_job.transform([Job_Title])[0]
    sample=[[Age,Gender,Education_Level,Job_Title,Years_of_Experience]]
    result=log.predict(sample)[0]
    st.success(f"Expected Avrege salary is : {round(result)}")