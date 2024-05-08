import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('plant_detection_model_2.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
 
# Function to make predictions
def predict_plant(image_path, class_indices):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = list(class_indices.keys())[predicted_class]
    return predicted_label, prediction[0][predicted_class]

# Example usage
input_image_path = 'orange.jpg'
# Define your class indices dictionary here based on your training data
class_indices = {
    'aloevera': 0, 'banana': 1, 'bilimbi': 2, 'cantaloupe': 3, 'cassava': 4, 'coconut': 5, 'corn': 6, 'cucumber': 7,
    'curcuma': 8, 'eggplant': 9, 'galangal': 10, 'ginger': 11, 'guava': 12, 'kale': 13, 'longbeans': 14, 'mango': 15,
    'melon': 16, 'orange': 17, 'paddy': 18, 'papaya': 19, 'peper chili': 20, 'pineapple': 21, 'pomelo': 22,
    'shallot': 23, 'soybeans': 24, 'spinach': 25, 'sweet potatoes': 26, 'tobacco': 27, 'waterapple': 28,
    'watermelon': 29
}
predicted_label, confidence = predict_plant(input_image_path, class_indices)
print('Predicted plant:', predicted_label)
print('Confidence:', confidence)