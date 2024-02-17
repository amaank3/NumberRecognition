import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import sys

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    loss = np.sum(log_probs) / m
    return loss

def gradient_descent(params, grads, learning_rate):
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
    return params

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

input_size = x_train_flat.shape[1]
hidden_size = 128
output_size = num_classes
learning_rate = 0.01
num_epochs = 10
batch_size = 64
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)
output_file = open('output.txt', 'w')
original_stdout = sys.stdout
sys.stdout = output_file
losses = []
for epoch in range(num_epochs):
    permutation = np.random.permutation(x_train_flat.shape[0])
    x_train_shuffled = x_train_flat[permutation]
    y_train_shuffled = y_train_onehot[permutation]
    for i in range(0, x_train_flat.shape[0], batch_size):
        x_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        hidden_layer = relu(np.dot(x_batch, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        probs = softmax(scores)
        loss = cross_entropy_loss(probs, y_batch)
        losses.append(loss)
        dscores = probs - y_batch
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0)
        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden_layer <= 0] = 0
        dW1 = np.dot(x_batch.T, dhidden)
        db1 = np.sum(dhidden, axis=0)
        
        W1, b1, W2, b2 = gradient_descent([W1, b1, W2, b2], [dW1, db1, dW2, db2], learning_rate)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png') 
plt.close()

sys.stdout = original_stdout
output_file.close()

image_path = 'num.png' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_resized = cv2.resize(image, (28, 28))
image_normalized = image_resized / 255.0
image_flat = image_normalized.reshape(1, -1)
hidden_layer = relu(np.dot(image_flat, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
probs = softmax(scores)
predicted_digit = np.argmax(probs)
print("Predicted digit:", predicted_digit)
