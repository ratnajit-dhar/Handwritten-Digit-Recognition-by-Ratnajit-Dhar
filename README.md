# Handwritten-Digit-Recognition-by-Ratnajit-Dhar
The app is deployed in the following link:
[Handwritten Digit Recognition by Ratnajit Dhar](https://handwritten-digit-recognition-by-ratnajit-dhar.streamlit.app/)

**Model Architecture:**
---

I have trained the dataset using a Convolutional Neural Network (CNN) model, which is particularly effective for image recognition tasks. Here is a breakdown of the model architecture:

1. **Conv2D Layer:** The first layer is a convolutional layer with 16 filters, each of size 3x3. This layer uses the ReLU activation function and takes an input shape of (28, 28, 1), which corresponds to the 28x28 pixel grayscale images from the MNIST dataset.

2. **BatchNormalization Layer:** This layer normalizes the activations of the previous layer, improving the training speed and stability of the model.

3. **MaxPooling2D Layer:** This pooling layer reduces the spatial dimensions (height and width) of the feature maps by taking the maximum value in each 2x2 pool. This helps in reducing the computational complexity and preventing overfitting.

4. **Dropout Layer:** This layer randomly sets 25% of the input units to 0 at each update during training, which helps prevent overfitting.

5. **Conv2D Layer:** The second convolutional layer has 32 filters, each of size 3x3, and uses the ReLU activation function.

6. **BatchNormalization Layer:** Another batch normalization layer to stabilize and speed up training.

7. **MaxPooling2D Layer:** Another pooling layer to further reduce the spatial dimensions of the feature maps.

8. **Dropout Layer:** Another dropout layer to further prevent overfitting.

9. **Flatten Layer:** This layer flattens the 2D feature maps into a 1D vector, preparing it for the fully connected layers.

10. **Dense Layer:** A fully connected layer with 64 units and ReLU activation function, adding more complexity to the model.

11. **BatchNormalization Layer:** Another batch normalization layer to stabilize and speed up training.

12. **Dropout Layer:** Another dropout layer to prevent overfitting.

13. **Dense Layer:** The final output layer with 10 units and a softmax activation function, which outputs the probabilities for each of the 10 classes (digits 0-9).


Thanks to this [repository](https://github.com/Vinay10100/Handwritten-Digit-Recognition/tree/main) by Vinay10100 for helping me with the implementation of st_canvas
