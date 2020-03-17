import re
import nltk
import emoji
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input

data = pd.read_csv('emojify_data.csv', delimiter=',', header=None)

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
    
def label_to_emoji(label):
    
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

index = np.random.randint(0, len(data[0])+1)
print(data[0][index], label_to_emoji(data[1][index]))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in set(nltk.corpus.stopwords.words('english'))]
    text = ' '.join(text)
    return text

corpus = []
for text in tqdm(data[0]):
    text = clean_text(text)
    corpus.append(text)

embedding_dims = 50
vocab_size = 246
max_len = 64
batch_size = 32
epochs = 50

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

x_train = pad_sequences(sequences, max_len, padding='post', truncating='post')
y_train = data.iloc[:, 1].values
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)

del data
del corpus
del sequences
del text

embedding_dict={}
with open('glove.6B.50d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict[word] = vectors
f.close()

embedding_matrix = np.zeros((vocab_size, embedding_dims))
for word, i in tqdm(tokenizer.word_index.items()):
    if i < vocab_size:
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

del embedding_dict
del embedding_vector

def Emojify_model():
    x_in = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dims, input_length=max_len)(x_in)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    x_out = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=x_in, outputs=x_out)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = Emojify_model()

model.summary()

model.layers[1].set_weights([embedding_matrix])

history = model.fit(x_train, y_train, epochs = 50, batch_size = 32, validation_split=0.25)

del x_train
del y_train

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracies')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('Losses')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

original_text = "I'm so Happy and Excited"
text = clean_text(original_text)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
x_pred = pad_sequences(sequences, max_len, padding='post', truncating='post')

y_pred = np.argmax(model.predict(x_pred))

print('Sentense:', original_text, 'Prediction:', label_to_emoji(y_pred))

import gc
gc.collect()
