import streamlit as st
import numpy as np
import pickle


samp = pickle.load(open('data.pkl','rb'))

st.title('Predict Credit Score')

st.write(samp.head())
sample = pickle.load(open('final.pkl','rb'))

group = st.selectbox("Select City", samp['group'].unique())
if group == "Hyderabad":
    group = 0
elif group == "Gurugram":
    group = 1
else:
    group = 2

order_status = st.selectbox("Order Status", samp['order_status'].unique())
if order_status == "processed":
    order_status = 1
elif order_status == "accepted":
    order_status = 1
elif order_status == "delivered":
    order_status = 1
elif order_status == "cancelled":
    order_status = -1
elif order_status == "new":
    order_status = 1
elif order_status == "rejected":
    order_status = -1
else:
    order_status = 1

retailer_names = st.selectbox("Retailer", sample['retailer'].unique())


value = st.number_input("Value of product")

model = pickle.load(open('model.pkl','rb'))

if st.button('Predict'):

    re = np.array([order_status,value,group,retailer_names])
    re = re.reshape(1,4)
 
    result=model.predict(re)
    st.header("Credit Score")
    st.write(str(result))
