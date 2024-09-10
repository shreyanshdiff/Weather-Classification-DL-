# mlflow_logging.py
import mlflow

def log_rnn_model(rnn_model, scaler, temp, dew_point_temp, rel_hum, wind_dir, wind_spd, stn_press):
    with mlflow.start_run():
        mlflow.log_param("Temperature", temp)
        mlflow.log_param("Dew Point Temperature", dew_point_temp)
        mlflow.log_param("Relative Humidity", rel_hum)
        mlflow.log_param("Wind Direction", wind_dir)
        mlflow.log_param("Wind Speed", wind_spd)
        mlflow.log_param("Station Pressure", stn_press)
        # Log the model
        mlflow.keras.log_model(rnn_model, "rnn_model")
        mlflow.log_artifact("scaler.pkl")
        print("RNN model and parameters logged successfully.")

def log_cnn_model(cnn_model, img_height, img_width):
    with mlflow.start_run():
        mlflow.log_param("Image Height", img_height)
        mlflow.log_param("Image Width", img_width)
        # Log the model
        mlflow.keras.log_model(cnn_model, "cnn_model")
        print("CNN model and parameters logged successfully.")
