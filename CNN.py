import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Specify the path to dataset
dataset_path = r"C:\Users\USER\OneDrive\Desktop\Image-Classification-main\Dataset_Celebrities\cropped"
AUTOTUNE = tf.data.AUTOTUNE
# Create train and validation generators
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # Adjust the output size based on the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=validation_generator, epochs=10)



print("--------------------------------------\n")
# model.summary()
print("--------------------------------------\n")



# Make predictions on the validation set
predictions = model.predict(validation_generator)

# Decode one-hot encoded predictions to class labels
predicted_classes = predictions.argmax(axis=1)

# Get the true class labels
true_classes = validation_generator.classes

# Evaluate the model on the validation set
accuracy = model.evaluate(validation_generator)[1]
print(f'Validation Accuracy: {accuracy * 100:.2f}%')


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']




# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'C:\Users\USER\OneDrive\Desktop\Image-Classification-main\plots\accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'C:\Users\USER\OneDrive\Desktop\Image-Classification-main\plots\loss_plot.png')





# Save the trained model
model.save(r'C:\Users\USER\OneDrive\Desktop\Image-Classification-main\Model')