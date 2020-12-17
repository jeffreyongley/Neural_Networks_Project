import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_labels[6])
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #10 Nueronss will represent the 10 classes

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([ #create a sequence of layers
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"), #create a fully connected layer with 128 nuerons with act. function rectify linear unit
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5) #creation of model


prediction = model.predict(test_images) #returns the values of the nuerons need to find the highest value there and set it as the classifier the argmax function of numpy does this
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("tested ac: ", test_acc)
