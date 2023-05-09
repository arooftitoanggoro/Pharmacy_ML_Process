# Pharmacy_ML_Process

The project aims to predict the potential profit of each medicine sold in Indonesia by analyzing the sales data and identifying the key factors that influence the profit potential using machine learning techniques, specifically decision tree and random forest algorithms. This is important because the pharmaceutical industry in Indonesia is rapidly growing, and predicting the profit potential of each medicine sold can be challenging due to the diverse geography and regional differences.

The proposed approach to solve this problem is to collect and clean the sales data, preprocess it, and analyze it using decision tree and random forest algorithms. The models will be trained, validated, and optimized to evaluate their performance using Mean Absolute Error (MAE) as the metric. The lower the MAE, the better the model's accuracy, and a successful model should have a low MAE value, indicating accurate predictions.

The dataset used in this project is a sample internal company dataset with masking data that consists of various features such as the medicine name, distribution channels, and regional information. The target variable is the profit potential of each medicine sold in Indonesia. The two machine learning algorithms, Decision Tree and Random Forest, will be used to predict the profit potential of each medicine sold in Indonesia. Decision Tree is a supervised learning algorithm that builds a tree-like model of decisions, and Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust model.

https://medium.com/@aroof.tito/pharmacy-sales-profit-machine-learning-process-6f2a0d445392

# The Training and Predict Scripts

Run the following code in the terminal to run the training and testing process
python src/data_pipeline.py

python src/preprocessing.py

python src/modeling.py

To run locally, run the following script:
sudo docker composition

# Flow Chart Model

<p align="center">
  <img src="https://github.com/arooftitoanggoro/Pharmacy_ML_Process/blob/master/assets/ML%20Process%20Pharmacy%20(chart).jpg" width="350" title="hover text">
</p>

The following is a block diagram consisting of:

Block diagram preparation (data_pipeline.py) - Green

Block diagram preprocessing & features engineering (preprocessing.py) - Red

Modeling and evaluation block diagram (modeling.py) - Blue


# Message format to make predictions via the API and response from the API

```
  from fastapi import FastAPI
  from pydantic import BaseModel
  import uvicorn
  import pandas as pd
  import util as util
  import data_pipeline as data_pipeline
  import preprocessing as preprocessing

  config_data = util.load_config()
  le_encoder = util.pickle_load(config_data["le_encoder_path"])
  model_data = util.pickle_load(config_data["production_model_path"])

  class api_data(BaseModel):
      Ship_Mode : str
      Customer_ID : str
      Customer_Name : str
      Segment : str
      Country : str
      City : str
      State : str
      Region : str
      Product_ID : str
      Category : str
      Sub_Category : str
      Product_Name : str
      Sales : float
      Quantity : float
      Discount : float

  app = FastAPI()

  @app.get("/")
  def home():
      return "Hello, FastAPI up!"

  @app.post("/predict/")
  def predict(data: api_data):    
      # Convert data api to dataframe
      data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

      # Convert dtype
      data = pd.concat(
          [
              data[config_data["predictors"][0]],
              data[config_data["predictors"][1:11]].astype(int),
              data[config_data["predictors"][11:]].astype(float)
          ],
          axis = 1
      )

      # Check range data
      try:
          data_pipeline.check_data(data, config_data, True)
      except AssertionError as ae:
          return {"res": [], "error_msg": str(ae)}

      # Encoding
      data = preprocessing.le_transform(data, le_encoder)

      # Predict data
      y_pred = model_data["model_data"]["model_object"].predict(data)

      # Inverse tranform
      y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

      return {"res" : y_pred, "error_msg": ""}

  if __name__ == "__main__":
      uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
```      
# Format Message streamlit.py

```
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
      else: st.warning("Prediksi Profit: res["res"] ")

 ```
 
 Input data requirement for API
 
      Ship_Mode : Standard Class
      Customer_ID : PJ-18835
      Customer_Name : Hospital-174
      Segment : Hospital
      Country : Indonesia
      City : Sumatera Utara-431
      State : Sumatera Utara
      Region : Sumatera
      Product_ID : AC-10004877
      Category : Diabet
      Sub_Category : Diabet/A
      Product_Name : Diabet/AAC-10004877
      Sales : 27.6
      Quantity : 4
      Discount : 0

Response data of prediction
      ```
      Prediksi Profit: 5.208
      ```
      
