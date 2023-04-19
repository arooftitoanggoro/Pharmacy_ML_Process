import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Pharmacy Sales Profit Prediction")
st.subheader("Just enter variabel below then click Predict button :")

# Create form of input
with st.form(key = "sales_data_form"):
    # Create select box input
    ship_mode = st.selectbox(
        label = "1.\tType of delivery:",
        options = (
            "First Class"
            "Same Day"
            "Second Class"
            "Standard Class"
        )
    )

    customer_id = st.text_input(
        label = "2.\tEnter customer id:")
    
    customer_name = st.text_input(
        label = "3.\tEnter customer name:")   

    # Create box for number input
    segment = st.selectbox(
        label = "4.\tEnter Segment:",
        options = (
            "Hospital"
            "Pharmacy"
            "Retail"
        )
    )

    country = st.text_input(
        label = "5.\tEnter country:")

    city = st.text_input(
        label = "6.\tEnter City:")
    
    state = st.text_input(
        label = "7.\tEnter State:")

    region = st.selectbox(
        label = "8.\tEnter Region:",
            options = (
            "Bali Nusa"
            "Jawa"
            "Kalimantan"
            "Papua"
            "Sulawesi"
            "Sumatera"
        )
    )
    
    product_id = st.text_input(
        label = "9.\tEnter Product ID:")
    
    category = st.selectbox(
        label = "10.\tEnter Category:",
            options = (
            "Diabet"
            "Hyper"
            "Respi"
        )
    )
    
    sub_category = st.selectbox(
        label = "11.\tEnter Sub-Category:",
            options = (
            "Diabet/A"
            "Diabet/B"
            "Diabet/C"
            "Diabet/D"
            "Hyper/A"
            "Hyper/B"
            "Hyper/C"
            "Hyper/D"
            "Hyper/E"
            "Hyper/F"
            "Hyper/G"
            "Hyper/H"
            "Hyper/I"
            "Respi/A"
            "Respi/B"
            "Respi/C"
            "Respi/D"
        )
    )

    product_name = st.text_input(
        label = "12.\tEnter Product Name:")

    sales = st.number_input(
        label = "13.\tEnter Sales Value in USD:",
        min_value = -99999,
        max_value = 99999,
    )

    quantity = st.number_input(
        label = "14.\tEnter Sales Quantity:",
        min_value = -99999,
        max_value = 99999,
    )
    
    discount = st.number_input(
        label = "15.\tEnter Sales discount:",
        min_value = -99999,
        max_value = 99999,
    )
    
# Create button to submit the form
    submitted = st.form_submit_button("Predict")

# Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "Ship_Mode": ship_mode,
            "Customer_ID": customer_id,
            "Customer_Name": customer_name,
            "Segment": segment,
            "Country": country,
            "City": city,
            "State": state,
            "Region": region,
            "Product_ID": product_id,
            "Category": category,
            "Sub-Category": sub_category,
            "Product_Name": product_name,
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount
        }

# Create loading animation while predicting
    with st.spinner("Sending data to prediction server ..."):
        res = requests.post("http://api:8080/predict", json = raw_data).json()

# Parse the prediction result
    if res["error_msg"] != "":
        st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
    else:
        if res["res"] != "Ya":
            st.warning("Prediksi Churn Customer: Tidak.")
        else:
            st.success("Prediksi Churn Customer: Ya.")