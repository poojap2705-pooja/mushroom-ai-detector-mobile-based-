import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pandas as pd
import sys
import pickle

# Load the model
model = tf.keras.models.load_model('mushroom_model.h5')

# Load class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Image parameters
img_height, img_width = 150, 150

# Load Excel data
excel_file = 'mushroom_dataset (A-X).xlsx'
df = pd.read_excel(excel_file)

# Function to classify image
def classify_mushroom(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    mushroom_name = class_names[predicted_class_index]

    # Get details
    details = df[df['name'] == mushroom_name].to_dict('records')[0]

    return details

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    result = classify_mushroom(img_path)
    print("Mushroom Details:")
    for key, value in result.items():
        print(f"{key}: {value}")