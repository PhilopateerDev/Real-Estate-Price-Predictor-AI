import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- إعدادات الصفحة ---
st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Prediction Web App")

# --- خطوة قراءة البيانات ---
df = pd.read_csv('https://raw.githubusercontent.com/PhilopateerDev/My-Projects/main/Real-Estate-Price-Predictor-AI/housing_data.csv')

st.subheader("Dataset Preview")
st.write(df.head())

# --- معالجة عمود المكان (Location) ---
if df['Location'].dtype == 'object':
    locations_list = df['Location'].unique().tolist()
    location_mapping = {name: i for i, name in enumerate(locations_list)}
    df['Location'] = df['Location'].map(location_mapping)
    display_locations = locations_list
else:
    display_locations = None

# --- تقسيم البيانات إلى X و y ---
X = df.drop('price', axis=1)
y = df['price']

# --- تقسيم البيانات لتدريب واختبار ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- تجهيز البيانات (Scaling) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- تدريب الموديل (Random Forest) ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- حساب التوقعات والدقة ---
y_pred = model.predict(X_test_scaled)
score = r2_score(y_test, y_pred)

# --- الرسم البياني ---
st.subheader("Prediction Accuracy Chart")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='green', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
st.pyplot(fig)

# --- وضع الـ R2 Score تحت الرسمة مباشرة مع علامة صح ---
# استخدمنا st.success عشان تظهر بخلفية خضراء وعلامة صح
st.success(f"✅ Model Accuracy : {score*100:.2f}%")

# --- مدخلات المستخدم (تحت الدقة وقبل التوقع) ---
st.divider()
st.subheader("Enter House Details for Prediction:")

area = st.number_input("Area in Square Meters", min_value=0.0)
rooms = st.number_input("Number of Rooms", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)

if display_locations:
    selected_loc = st.selectbox("Location", options=display_locations)
    location_val = location_mapping[selected_loc]
else:
    location_val = st.number_input("Location Code", min_value=0)

# --- زر التوقع ---
if st.button("Predict Price Now"):
    user_input = np.array([[area, rooms, bathrooms, location_val]])
    user_input_scaled = scaler.transform(user_input)
    final_prediction = model.predict(user_input_scaled)
    st.info(f"Estimated Market Price: ${final_prediction[0]:,.2f}")
    
