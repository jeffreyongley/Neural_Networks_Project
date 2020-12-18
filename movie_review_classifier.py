import tensorflow as tf
from tensorflow import keras
import numpy as np
np.set_printoptions(suppress=True)

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) #only take the 10000 most common words


word_index = data.get_word_index() # gives dictionary that gives int: word mappings in tuples

word_index = {k:(v+3) for k, v in word_index.items()} 
word_index["<PAD>"] = 0 # USed to pad reviews with 0's to make all revies the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #swaps from (word, integer) to (integer, word)


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)



def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) # returns the human readable words we want
'''
# Define Model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) #model.add adds layers // The point of embedding layer is to take word vectors and group them based of similar and different words
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) # sq3uishes to between 0 and 1

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # loss function takes the value which sigmoid function gives as between 0 and 1 and finds how far it is from 0 and 1

#validdation data checks how well model is performing based on tunes and tweaks we are making on training data on new data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000] # cut train data down to 10000 values
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
# 7 MIN MARK OF VIDEO 7/9

model.save("model.h5")
'''

def review_encode(s):
    encoded = [1] # set start tag for consistency

    for word in s:
        if word.lower() in word_index: # if word exists in know words append it
            encoded.append(word_index[word.lower()])
        else: #otherwise use unknown tag
            encoded.append(2)

    return encoded


model = keras.models.load_model("model.h5")

with open("lion_king_review.txt", encoding ="utf-8") as f:
    for line in f.readlines(): # Go line by line
        nline = line.replace(",", "").replace(".","").replace(":", "").replace("(", "").replace(")","").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


'''
print(train_data[0])
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''
