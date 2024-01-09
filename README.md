# Image-Classification

This code implements a Convolutional Neural Network (CNN) for image classification.
The CNN consists of two convolutional layers with ReLU activation functions, followed by MaxPooling layers.
The output is flattened and fed into a dense layer with ReLU activation, and finally another dense layer with softmax activation for classification.
The model takes images of size 224x224x3 as input and outputs a probability distribution over 5 classes.
The optimizer used is Adam and the loss function is categorical cross-entropy.


## Training Process:

* The dataset is loaded using ImageDataGenerator with rescaling and validation split.
* Separate generators are created for training and validation data with a batch size of 32.
* The model is trained for 10 epochs and the accuracy and loss are monitored.
* The model is evaluated on the validation data after each epoch.
* The training accuracy and validation accuracy are plotted over time.
* The loss on both the training and validation set is also plotted over time.

## Results:
* The model achieves a validation accuracy of 87.10% after 10 epochs.
* The training and validation accuracy curves show that the model is learning and generalizing well and the plots are attached the repository.
* The loss curves show that the model is able to optimize the loss function effectively.
* The model is successfully trained and can be used to classify images into 5 different classes.
