# 🏠 Real Estate Price Predictor AI

A professional Machine Learning web application that predicts house prices with **99% accuracy**. Built using Python, Scikit-Learn, and Streamlit.

## 🚀 Live Demo
You can try the live application here: 
**[https://real-estate-price-predictor-ai-g7ep9y9gwqvbmrswuxszf5.streamlit.app/](https://real-estate-price-predictor-ai-g7ep9y9gwqvbmrswuxszf5.streamlit.app/)**

## 🛠️ Features
* **Model**: Powered by `RandomForestRegressor` for robust and accurate predictions.
* **Accuracy**: Achieved a high **R2 Score of 0.99**.
* **Preprocessing**: Includes automated `StandardScaler` for feature scaling and Label Encoding for locations.
* **User Interface**: Clean and interactive web dashboard built with `Streamlit`.
* **Data Visualization**: Features a "Prediction Accuracy Chart" to compare Actual vs. Predicted values.

## 📊 How to Use
1. Enter the **Area** of the house in square meters.
2. Specify the number of **Rooms** and **Bathrooms**.
3. Select the **Location** from the dropdown menu.
4. Click **"Predict Price Now"** to get the estimated market value.

## 📁 Dataset
The model is trained on a custom dataset (`housing_data.csv`) containing real estate information including Area, Rooms, Bathrooms, and Locations (like New Cairo, Sheikh Zayed, Maadi, etc.).

## 💻 Tech Stack
* **Language**: Python
* **ML Library**: Scikit-Learn
* **Web Framework**: Streamlit
* **Visualization**: Matplotlib
* **Data Handling**: Pandas & Numpy

## 🛠️ Local Installation
If you want to run this project locally:
1. Clone the repository:
   ```bash
   git clone [https://github.com/PhilopateerDev/My-Projects.git](https://github.com/PhilopateerDev/My-Projects.git)
2.Install dependencies:
pip install -r requirements.txt
3.Run the app:
streamlit run app.py
