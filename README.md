# MNIST Classification with Artificial Neural Networks (ANN)

This project demonstrates the classification of the MNIST dataset (handwritten digits) using an Artificial Neural Network (ANN) built with Keras and TensorFlow.

## Project Overview

- **Dataset**: MNIST dataset from `tensorflow.keras.datasets`
- **Algorithm**: Artificial Neural Networks (ANN)
- **Objective**: Classify handwritten digits (0-9) based on pixel values.
- **Activation Functions**: Sigmoid, ReLU, Softmax
- **Tools Used**: Python, Keras, TensorFlow, Matplotlib, Seaborn

## Steps:
1. Load and preprocess the MNIST dataset.
2. Train a simple ANN model using the sigmoid activation function.
3. Evaluate the model using accuracy, confusion matrix, and classification report.
4. Build a deeper model with ReLU activation functions to improve accuracy.
5. Train and evaluate the deeper model with more layers and ReLU activations.

## Results:
- **Accuracy (Simple Model)**: Achieved ~92% accuracy with a simple ANN model.
- **Accuracy (Deeper Model)**: Achieved ~98% accuracy with a deeper ANN model.
- **Confusion Matrix**: Shows the number of correct and incorrect classifications.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

## Why Sigmoid and ReLU?
- **Sigmoid**: Used in the initial simple model for classification. While it works, it's prone to the vanishing gradient problem in deeper networks.
- **ReLU**: Used in deeper layers for better performance and faster convergence. It handles the vanishing gradient problem and improves the network's performance.
- **Softmax**: Applied in the output layer for multi-class classification, as it converts the outputs to probabilities for each class.

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mnist_classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mnist_classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `mnist_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal:
    ```bash
    python mnist_classification.py
    ```

## License
This project is licensed under the MIT License.
