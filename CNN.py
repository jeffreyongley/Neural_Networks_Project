# Adapted from Tensorflow tutorial
# Utilizes Convolutional Neural Network to classify CIFAR images 
# this process utilizes Keras Sequential API

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
'''
Show first 25 images and labels
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #CIFAR labels happen to be arrays hence the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
'''
#def define_model(): # Fuction that initially defines model and saves in file CNN.h5
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
# current output of each layer is a 3-D Tensor of shape (height, width, channels)
# to complete model we must feed into one or more dense layers
# input of dense layers is a vector (1-D) must flatten first
# our data set has 10 output classes so the final dense layer must have 10 outputs

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10))

#model.summary()
#compile and train the model
model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

#model.save("CNN.h5")

#def evaluate_model(filename):
#model = load_model(filename)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
'''
def main():
    evaluate_model("CNN.h5")

if __name__ == "__main__":
    main()
'''