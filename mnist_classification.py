# MNIST Classification using Artificial Neural Networks (ANN)

# Import necessary libraries
import tensorflow as tf  # type: ignore
from tensorflow import keras # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # type: ignore

# Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data (values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the data (28x28 images to 784-dimensional vectors)
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

# Define the ANN model with one hidden layer and sigmoid activation
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')  # Output layer with 10 neurons (for 10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train_flattened, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)

print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted_labels))

# Build a deeper model with multiple hidden layers and different activation functions
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # First hidden layer with ReLU activation
    keras.layers.Dense(200, activation='relu'),                      # Second hidden layer with ReLU activation
    keras.layers.Dense(10, activation='softmax')                     # Output layer with 10 neurons and softmax
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train_flattened, y_train, epochs=10)

# Evaluate the updated model
test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)
print(f"\nTest accuracy with deeper model: {test_accuracy * 100:.2f}%")

# Make predictions with the updated model
y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Plot confusion matrix for updated model
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report for updated model
print(classification_report(y_test, y_predicted_labels))
