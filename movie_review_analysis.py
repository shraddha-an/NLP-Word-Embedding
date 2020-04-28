# NLP Sentiment Analysis of Movie Reviews

# Importing libraries
import pandas as pd, numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

from os import listdir
from collections import Counter

# Creating a class to bundle all the data preprocessing functions
class DataPreprocessing():
    # Function to read the content of files
    def load_doc(self, file):
        file = open(file, 'r')
        text = file.read()
        file.close()
        return text

    # Func to clean up text data
    def clean_doc(self,text):
        main_words = re.sub('[^a-zA-Z]', ' ', text)
        main_words = (main_words.lower()).split()
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]
        main_words = [w for w in main_words if len(w) > 1]
        main_words = ' '.join(main_words)
        return main_words

    # Func to update tokens to vocab
    def add_2_vocab(self, file, vocab):
        doc = self.load_doc(file)
        tokens = self.clean_doc(doc)
        vocab.update(tokens)

    # Func to iterate throgh all files in the directory
    def process_docs(self, directory, vocab, isTrain):
        documents = []
        for file in listdir(directory):
            # Skip files starting with cv9 --> test set files
            if isTrain and file.startswith('cv9'):
                continue
            if not isTrain and not file.startswith('cv9'):
                continue
            path = directory + '/' + file

            # Call the documents & load their content
            doc = self.load_doc(path)
            token = self.clean_doc(doc)
            documents.append(token)
        return documents

# Keeping only those words that appear at least twice in the vocab
vocab = Counter()
min_occurence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurence]

# Saving the list of tokens in a txt file
def save_list(tokens, filename):
    data = '\n'.join(tokens)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# Creating an object of the Data Preprocessing class
clean_data = DataPreprocessing()

# Creating the training corpus
positive_reviews = clean_data.process_docs('txt_sentoken/pos', vocab, True)
negative_reviews = clean_data.process_docs('txt_sentoken/neg', vocab, True)
train = positive_reviews + negative_reviews

# Tokenizing & Encoding the vocab to create vector representation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train)
encoded = tokenizer.texts_to_sequences(train)

# Creating X_train & y_train
k = max([len(s.split()) for s in train])
X_train = pad_sequences(sequences = encoded, maxlen = k, padding = 'post')
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

# Creating Testing docs
positive_reviews = clean_data.process_docs('txt_sentoken/pos', vocab, False)
negative_reviews = clean_data.process_docs('txt_sentoken/neg', vocab, False)
test = positive_reviews + negative_reviews

encoded = tokenizer.texts_to_sequences(test)

X_test = pad_sequences(sequences = encoded, maxlen = k, padding = 'post')
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

# Vocab size
vocab_size = len(tokenizer.word_counts) + 1

# Building the CNN model
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

model = Sequential()

model.add(Embedding(vocab_size, 100, input_length = k))
model.add(Conv1D(filters = 32, kernel_size = 8, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, verbose = 1)

# Predicting on Test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) # Converting probabilites to 0/1 values


# Evaluating the model's accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
