import streamlit as st
import pickle
import numpy as np

lin=pickle.load(open('lin_model.pkl','rb'))
dt = pickle.load(open('dec_tree_model.pkl','rb'))
rf= pickle.load(open('rand_for_model.pkl','rb'))
xgb = pickle.load(open('xgb_model.pkl','rb'))
ada = pickle.load(open('ada_reg_model.pkl','rb'))

st.title("Medical Premium prediction Web App")
html_temp = """
    <div style="background-color:lightgreen ;padding:8px">
    <h2 style="color:black;text-align:center;">Medical premium based on health problems</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

activities = ['Linear Regression','Decision Tree','Random Forest','AdaBoost']
option = st.sidebar.selectbox('Which regression model would you like to use?',activities)
st.subheader(option)

st.write("""###### For selction:
            0: NO , 1: YES """)

diabetes = [0,1]
diabetes_option = st.sidebar.selectbox("Do you have diabetes? ",diabetes)
diabetes_option=float(diabetes_option)

bp = [0,1]
bp_option = st.sidebar.selectbox("Do you have bloop pressure problems?",bp)
bp_option=float(bp_option)

transplant = [0,1]
trans_option = st.sidebar.selectbox("Have you undergone any transplant",transplant)
trans_option=float(trans_option)

kaller = [0,1]
kaller_option = st.sidebar.selectbox("Do you have any known allergies?",kaller)
kaller_option=float(kaller_option)

cdisease = [0,1]
cdisease_option = st.sidebar.selectbox("Do you have any chronic disease?",cdisease)
cdisease_option=float(cdisease_option)

cancer = [0,1]
cancer_option = st.sidebar.selectbox("Does any of your family member has a history of cancer?",cancer)
cancer_option=float(cancer_option)

age = st.slider('Chose your age',18, 60)
weight = st.slider('Chose your weight', 40, 95)
height = st.slider('Chose your height in centimeters', 130, 180)
surgeries = st.slider('Chose number of surgeries you have undergone', 0,2)

inputs=[[age,diabetes_option,bp_option,trans_option,cdisease_option,height,weight,kaller_option,cancer_option,
surgeries]]

if st.button('Predict'):
    if option=='Linear Regression':
        st.success(lin.predict(inputs))
    elif option=='Decision Tree':
        st.success(dt.predict(inputs))
    elif option=='Random Forest':
        st.success(rf.predict(inputs))
    # elif option=='XGBoost':
    #     st.success(xgb.predict(inputs))
    else:
        st.success(ada.predict(inputs))

