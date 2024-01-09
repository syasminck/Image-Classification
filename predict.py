import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

# If you want to load the model in another file:
# Load the model
loaded_model = tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\Image-Classification-main\Model')

# Specify the path to the image you want to make predictions on
image_path = r'C:\Users\USER\OneDrive\Desktop\Image-Classification-main\Dataset_Celebrities\cropped\lionel_messi\lionel_messi1.png'
# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale pixel values

# Make predictions
predictions = loaded_model.predict(img_array)

# Decode the one-hot encoded predictions to class labels
predicted_class = np.argmax(predictions)

# Display the predicted class
#print(f'Predicted Class: {predicted_class}')




class_indices ={"Lionel Messi":0,"Maria Sharapova":1,"Roger Federer":2,"Serena Williams":3,"Virat Kohli":4}

# Invert the dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

# Get the predicted class name
predicted_class_name = class_names[predicted_class]

# Display the predicted class name
print(f'Predicted Class: {predicted_class_name}')