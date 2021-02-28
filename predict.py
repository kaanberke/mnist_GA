from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

X_train = X_train.astype('float32')
X_train /= 255
random_idx = np.random.randint(0, len(X_train))

model = load_model("cnn.h5")
output = model.predict(X_train[random_idx].reshape(1, *input_shape))
idx = np.argmax(output[0])
print("Model output: {0}".format(output))
print("Value with highest probability: {0} %{1:.2f}".format(idx, (output[0][idx]*100)))
