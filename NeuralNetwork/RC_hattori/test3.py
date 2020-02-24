from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os
import matplotlib.pyplot as plt

os.environ["http_proxy"] = "http://www-proxy.waseda.jp:8080"
os.environ["https_proxy"] = "https://www-proxy.waseda.jp:8080"

max_features = 10000
max_len = 500
batch_size = 32

print("loading data")
(input_train, y_train), (input_test, y_test) =\
    imdb.load_data(num_words=max_features)
print(len(input_train), "train sequences")
print(input_train)
print(len(input_test), "test sequences")
print("pad sequencs (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print("input_train shape: ", input_train.shape)
print(input_train)
print("input_test shape: ", input_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(input_train,   y_train, epochs=10, batch_size=128, validation_split=0.2)

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, label="training acc")
plt.plot(epochs, val_acc, label="validation acc")
plt.title("training and validation accuracy")
plt.show()