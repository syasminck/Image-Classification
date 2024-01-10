Image Classification

This code presents a Convolutional Neural Network (CNN) designed for image classification. 
The CNN architecture includes two convolutional layers with ReLU activation functions, followed by MaxPooling layers. 
The output is flattened and passed through a dense layer with ReLU activation, followed by another dense layer with softmax activation for classification. 
The model takes input images sized 224x224x3 and produces a probability distribution over 5 classes. 
The optimization is done using the Adam optimizer, and the categorical cross-entropy loss function is employed.

* Training Overview:

The dataset is loaded using ImageDataGenerator with rescaling and a validation split.
Separate generators are created for training and validation data, each with a batch size of 32.
The model undergoes training for 10 epochs, during which accuracy and loss metrics are tracked.
After each epoch, the model is evaluated on the validation data.
Training and validation accuracy are plotted over time.
The loss on both the training and validation sets is visualized over the training process.

* Results:

After 10 epochs, the model achieves a commendable validation accuracy of 87.10%.
Examination of the training and validation accuracy curves indicates that the model effectively learns and generalizes well. Refer to the attached repository for visualization plots.
The loss curves demonstrate the model's proficiency in optimizing the loss function.
In summary, the model is successfully trained and ready for classifying images into 5 distinct classes.
