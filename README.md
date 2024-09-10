

```markdown
# Weather Classification and Forecasting using ML and DL Frameworks

## Project Overview

This project involves weather classification and forecasting using a combination of Machine Learning (ML) and Deep Learning (DL) models. It also incorporates logging of the models and parameters using MLflow. The application is built using Streamlit and includes features like weather prediction using an RNN, weather classification using CNN and Random Forest, and potential chatbot interaction.

## Features

1. **RNN Weather Forecasting**:
   - Predicts weather conditions for the next 7 days using a Recurrent Neural Network (RNN) model.
   - Input data includes temperature, dew point temperature, relative humidity, wind direction, wind speed, and station pressure.
   - Visualizes the predictions over the 7-day period.

2. **CNN Weather Image Classification**:
   - Classifies weather images into various conditions using a pre-trained Convolutional Neural Network (CNN) model.
   - Supports image uploads in `.jpg`, `.jpeg`, and `.png` formats.
   - Displays the predicted weather condition and logs the CNN model and parameters using MLflow.

3. **Random Forest Weather Classification**:
   - Classifies weather conditions based on features like temperature, relative humidity, wind speed, visibility, and pressure using a Random Forest classifier.
   - Encodes categorical weather labels using a Label Encoder.
   - Logs the Random Forest model and parameters to MLflow.

4. **Weather API Integration**:
   - Uses the OpenWeatherMap API to fetch real-time weather data for a specified city.

5. **MLflow Logging**:
   - Logs the RNN, CNN, and Random Forest models along with their parameters for tracking and reproducibility.

## Project Structure

```bash
.
├── api.py                   # Contains API key and integration for OpenWeatherMap API
├── classification.pkl        # Trained RandomForestClassifier model for weather classification
├── CNN_Image.h5              # Trained CNN model for image classification
├── label_encoder.pkl         # Label encoder for classification model
├── log.py                    # Script for logging models and parameters with MLflow
├── scaler.pkl                # Scaler used for RNN model
├── weather_rnn.h5            # Trained RNN model for weather forecasting
├── streamlit_app.py          # Main Streamlit application script
└── README.md                 # This file
```

## Requirements

To run this project, install the following Python dependencies by creating a `requirements.txt` file and adding the following content:

```bash
streamlit==1.37.0
numpy
matplotlib
scikit-learn==1.4.2
keras
tensorflow
torch
requests
joblib==1.2.0
geopandas
plotly
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/shreyanshdiff/Weather-Classification-DL-.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Weather-Classification-DL-
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the provided URL (http://localhost:8501) in a web browser to interact with the application.

## Model Logging with MLflow

The project includes model logging functionality using MLflow. Ensure that you have MLflow set up on your system. Logs will be stored and tracked for the RNN, CNN, and RandomForest models.

- **RNN Weather Forecasting**: The model and parameters will be logged when you click the "Log RNN Model and Parameters" button.
- **CNN Weather Classification**: The CNN model will be logged along with input image parameters.
- **RandomForestClassifier**: This model and its parameters will be logged upon clicking "Log RF Model and Parameters."

## Pages

1. **About**: Displays the project title, student name, and guide name.
2. **RNN Weather Forecasting**: Provides 7-day weather forecasting based on user inputs using an RNN model.
3. **CNN Weather Classification**: Allows users to upload weather images and predicts weather conditions using a CNN model.
4. **Random Forest Weather Classification**: Predicts weather conditions based on input features using a RandomForestClassifier.

## Contributors

- **Name**: Shreyansh Singh (21BIT0604)
- **Guide**: Prof. Seetha R.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Instructions:

- Make sure to include the project name, description, setup instructions, and key features.
- In the `requirements.txt`, add the dependencies you used.
- If you have an API key for the weather chatbot, replace the placeholder `"API"` with the actual key or set up environment variables.
