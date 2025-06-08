import pandas as pd
import streamlit as st
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf                   
from tensorflow.keras.models import load_model                                                                                                                                                                                                     

# Load your merged dataset
df = pd.read_csv("merged_crop_data.csv")
df_final = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label', 'Season',
               'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
df_final = df_final.groupby(['label', 'State', 'Season'], as_index=False).mean(numeric_only=True)               

# Standardize column (remove whitespace, lowercase)
df_final['label'] = df_final['label'].str.strip().str.lower()
df_final['State'] = df_final['State'].str.strip().str.lower()
df_final['Season'] = df_final['Season'].str.strip().str.lower()
# Title
st.title("🌾 Crop Info Lookup")

# User input
crop_name = st.selectbox("Select Crop Name:", sorted(df_final['label'].unique()))
state_name = st.selectbox("Select State Name:", df_final['State'].unique())
season_name = st.selectbox("Select Season:", df_final['Season'].unique())

# Display the image
if crop_name == 'banana':
    image = Image.open('banana.jpg')
elif crop_name == 'rice':
    image = Image.open('rice.jpg')
elif crop_name == 'blackgram':
    image = Image.open('blackgram.jpg')
elif crop_name == 'chickpea':
    image = Image.open('chickpea.jpg')
elif crop_name == 'coconut':
    image = Image.open('Coconut+Tree.jpeg')
elif crop_name == 'cotton':
    image = Image.open('cotton.jpeg')
elif crop_name == 'jute':
    image = Image.open('jute.jpeg')
elif crop_name == 'lentil':
    image = Image.open('lentil.jpeg')
elif crop_name == 'kidneybeans':
    image = Image.open('kidneybeans.jpg')  
elif crop_name == 'maize':
    image = Image.open('maize.jpeg')
elif crop_name == 'mungbean':
    image = Image.open('mungbean.jpeg')
else:
    image = Image.open('Pigeon-Peas.jpg') 
image = image.resize((300, 300))  # Resize the image
st.image(image, caption=f"{crop_name.capitalize()} Image", use_column_width=True)                   

if crop_name and state_name:
    # Filter the DataFrame based on user input
    results = df_final[(df_final['label'] == crop_name) & (df_final['State'] == state_name) & (df_final['Season'] == season_name)]
    if not results.empty:
        st.success(f"Details found for crop: {crop_name.capitalize()}")
        st.write(f"State: {state_name.capitalize()}")
        st.write(f"Season: {season_name.capitalize()}")
        st.write("Here are the details:")
        results = results.rename(columns={
            'N': 'Nitrogen (ppm)',
            'P': 'Phosphorus (ppm)',
            'K': 'Potassium (pp)',
            'temperature': 'temperature (celsius)',
            'humidity': 'humidity (RH)',
            'ph': 'pH',
            'rainfall': 'rainfall (mm)',
            'Fertilizer': 'Fertilizer (kg/ha)',
            'Pesticide': 'Pesticide (kg/ha)',
            'Season': 'Season',
            'State': 'State',
            'label': 'Crop Name',
            'Annual_Rainfall': 'Annual Rainfall (mm)'
        })
        st.dataframe(results)
    else:
        if df_final[(df_final['label'] == crop_name) & (df_final['State'] == state_name)].empty:
            st.warning(f"No data found for crop: {crop_name.capitalize()} in state: {state_name.capitalize()}.")
        st.warning("Crop not found. Please check the spelling or try another crop.")


# Streamlit app for disease prediction in crops 
st.title("Disease Prediction in Crops")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

load_model = tf.keras.models.load_model('disease_recognition_model.keras')
img = Image.open(image).convert('RGB') # Ensure the image is in RGB format
img = img.resize((224, 224)) # Resize to the target size
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make a prediction
predictions = load_model.predict(img_array)

    # Get the predicted class
predicted_class_index = np.argmax(predictions)
class_names = list(train_generator.class_indices.keys()) # Get class names from your training generator
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class_name}")
