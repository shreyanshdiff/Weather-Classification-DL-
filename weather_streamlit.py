import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import geopandas as gpd
import plotly.express as px
import log  # Import the MLflow logging script
from api import API 
# Load your trained RNN model and scaler
rnn_model = load_model('weather_rnn.h5')
scaler = joblib.load('scaler.pkl')

# Load your trained CNN model
cnn_model = load_model('CNN_Image.h5')

# Load the classification model from classification.pkl
classification_model = joblib.load('classification.pkl')

# Load the RandomForestClassifier model
rf_model = joblib.load('classification.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define parameters for CNN
img_height, img_width = 150, 150

# Load pre-trained model and tokenizer for chatbot with left-padding
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# tokenizer.padding_side = 'left'
# chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to predict weather using RNN
def predict_weather(temp, dew_point_temp, rel_hum, wind_dir, wind_spd, stn_press):
    input_data = np.array([[temp, dew_point_temp, rel_hum, wind_dir, wind_spd, stn_press]])
    input_data = scaler.transform(input_data)
    input_sequence = np.tile(input_data, (3, 1))
    input_sequence = np.reshape(input_sequence, (1, 3, 6))
    prediction = rnn_model.predict(input_sequence)[0, 0]
    prediction = scaler.inverse_transform([[prediction, 0, 0, 0, 0, 0]])[0, 0]
    return prediction

# Function to preprocess the uploaded image for CNN
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make a prediction using the CNN classification model
def make_prediction_with_classification_model(img_array):
    img_array_flat = img_array.flatten().reshape(1, -1)
    prediction = classification_model.predict(img_array_flat)
    return prediction

# Function to get weather data using the chatbot
def get_weather(city):
    api_key = API  # Make sure to replace with your actual API key or use environment variables
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city + "&appid=" + api_key + "&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    
    if data.get("cod") != "404":
        if "main" in data and "weather" in data:
            main = data["main"]
            weather = data["weather"][0]
            temperature = main["temp"]
            description = weather["description"]
            return f"The temperature in {city} is {temperature}°C with {description}."
        else:
            return "Weather data not available."
    else:
        return "City Not Found"

# RNN Weather Forecasting Page
def rnn_weather_forecasting():
    st.title("Weather Forecasting Using RNN")
    
    with st.sidebar:
        temp = st.number_input("Temperature (°C)", value=-1.8)
        dew_point_temp = st.number_input("Dew Point Temperature (°C)", value=-3.9)
        rel_hum = st.slider("Relative Humidity (%)", 1, 100)
        wind_dir = st.slider("Wind Direction (°)", 1, 360)
        wind_spd = st.number_input("Wind Speed (km/h)", value=8.0)
        stn_press = st.number_input("Station Pressure (kPa)", value=101.24)
        st.warning("The application is not a production level code / implementation.")
    
    if st.button("Forecast"):
        predictions = []
        for day in range(1, 8):
            prediction = predict_weather(temp, dew_point_temp, rel_hum, wind_dir, wind_spd, stn_press)
            predictions.append(prediction)
            temp += np.random.normal(0, 1)
            dew_point_temp += np.random.normal(0, 1)
            rel_hum += np.random.normal(0, 1)
            wind_dir += np.random.normal(0, 1)
            wind_spd += np.random.normal(0, 1)
            stn_press += np.random.normal(0, 0.1)
        
        st.success(f"The predicted temperature for the next 7 days is: {predictions}")
        
        days = [f'Day {i}' for i in range(1, 8)]
        plt.figure(figsize=(10, 6))
        plt.plot(days, predictions, marker='o', linestyle='-', color='b', label='Predicted Temperature')
        plt.xlabel('Days')
        plt.ylabel('Temperature (°C)')
        plt.title('Predicted Temperature for Next 7 Days')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
        
    if st.button("Log RNN Model and Parameters"):
        log.log_rnn_model(rnn_model, scaler, temp, dew_point_temp, rel_hum, wind_dir, wind_spd, stn_press)
        st.success("RNN model and parameters logged successfully.")

# CNN Weather Classification Page
def cnn_weather_classification():
    st.title("Weather Classification Using Pre-trained CNN")
    
    uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file)
        img_array = preprocess_image(uploaded_file)
        
        # Use the classification model for prediction
        prediction = make_prediction_with_classification_model(img_array)
        
        st.write(f"Prediction: {prediction}")
        
        class_labels = ['snow', 'sandstorm', 'rime', 'rainbow', 'rain', 'lightning', 'hail', 'glaze', 'frost', 'fogsmog', 'dew']  
        
        if prediction[0] < len(class_labels):
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write(f"Predicted Weather Condition: {class_labels[prediction[0]]}")
        else:
            st.write(f"Error: Predicted class index {prediction[0]} is out of bounds for the class labels list.")
        
        if st.button("Log CNN Model and Parameters"):
            log.log_cnn_model(cnn_model, img_height, img_width)
            st.success("CNN model and parameters logged successfully.")

# Random Forest Weather Classification Page
def rf_weather_classification():
    st.title("Weather Classification Using RandomForestClassifier")
    
    # Input features for classification
    hour = st.slider("Hour of the day", 0, 23, 12)
    day = st.slider("Day of the month", 1, 31, 15)
    month = st.slider("Month of the year", 1, 12, 6)
    temp = st.number_input("Temperature (°C)", value=20.0)
    dew_point_temp = st.number_input("Dew Point Temperature (°C)", value=10.0)
    rel_hum = st.slider("Relative Humidity (%)", 0, 100, 50)
    wind_spd = st.number_input("Wind Speed (km/h)", value=5.0)
    visibility = st.number_input("Visibility (km)", value=10.0)  # Include Visibility
    press = st.number_input("Pressure (kPa)", value=101.0)  # Include Pressure
    
    if st.button("Classify"):
        # Prepare the input data
        input_data = np.array([[hour, day, month, temp, dew_point_temp, rel_hum, wind_spd, visibility, press]])
        
        # Make prediction
        prediction = rf_model.predict(input_data)[0]
        
        st.success(f"Predicted Weather Condition: {label_encoder.inverse_transform([prediction])[0]}")
        
        if st.button("Log RF Model and Parameters"):
            log.log_rf_model(rf_model, input_data)
            st.success("Random Forest model and parameters logged successfully.")

# # Weather Chatbot Page
# def weather_chatbot():
#     st.title("Weather Chatbot")
#     user_input = st.text_input("Enter your message:", key="chat_input")
#     if st.button("Send"):
#         if user_input:
#             new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
#             chatbot_model.chat_history_ids = torch.cat([chatbot_model.chat_history_ids, new_user_input_ids], dim=-1) if hasattr(chatbot_model, 'chat_history_ids') else new_user_input_ids
#             bot_output = chatbot_model.generate(chatbot_model.chat_history_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
#             bot_response = tokenizer.decode(bot_output[:, chatbot_model.chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
#             st.text_area("Bot:", bot_response, height=200, key="chatbot_response")
#         else:
#             st.write("Please enter a message.")

def about():
    st.title("PROJECT - 1")
    st.header("Weather Classification using ML and DL Frameworks and Logging using MLflow")
    st.subheader("NAME : SHREYANSH SINGH (21BIT0604)")
    st.subheader("GUIDE NAME : PROF SEETHA.R")
page = st.sidebar.selectbox("Choose a Page", ["About","RNN Weather Forecasting", "CNN Weather Classification", "Random Forest Weather Classification", "Weather Chatbot"])

if page =="About":
    about()
if page == "RNN Weather Forecasting":
    rnn_weather_forecasting()
if page =="CNN Weather Classification":
    cnn_weather_classification()
if page =="Random Forest Weather Classification":
    rf_weather_classification()
# if page =="Weather Chatbot":
#     weather_chatbot()
   
