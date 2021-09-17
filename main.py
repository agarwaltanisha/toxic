from flask import Flask, request, render_template, jsonify, json

import json
import codecs
import numpy as np

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import (
    Tokenizer, text_to_word_sequence, tokenizer_from_json
)
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import (
    Bidirectional, Conv1D, Embedding, LSTM, 
    Dropout, Dense, Input, GlobalMaxPooling1D
)

# #   CPU-ONLY
# import os
# import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
# session = tf.compat.v1.Session(config=configuration) 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#   PRE-PROCESSING, TOKENIZATION, AND PADDING 
def clean_text(seq):
    stopwords = open('resources/stopwords.txt','r').read().splitlines()
    tokens = text_to_word_sequence(
        seq, 
        filters="\"!'#$%&()*+,-˚˙./:;‘“<=·>?@[]^_`{|}~\t\n", 
        lower=True, 
        split=" "
    )
    cleanup = " ".join(filter(lambda word: word not in stopwords, tokens))
    return cleanup

def tokenize(comment, tokenizer):
    comment = list(clean_text(comment[0]))
    seq = tokenizer.texts_to_sequences(comment)
    padded_seq = pad_sequences(seq, maxlen=150)
    return padded_seq

#   PRE-TRAINED GLOVE EMBEDDINGS
def get_embeddings_matrix(tokenizer):
    vector_length=300
    length=150

    embeddings_index = {}
    f = codecs.open('resources/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    word_index = tokenizer.word_index
    nb_words = min(200000, len(word_index))
    notfound=[]
    embedding_matrix = np.zeros((nb_words, vector_length))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            notfound.append(word)

    return embeddings_index, embedding_matrix, nb_words

#   INITIALIZE MODEL
def get_model(tokenizer):

    vector_length=300
    length=150
    num_classes=6

    embeddings_index, embedding_matrix, nb_words = get_embeddings_matrix(tokenizer)

    inp = Input(shape=(length,))
    x = Embedding(nb_words, vector_length, weights=[embedding_matrix])(inp)
    x = Conv1D(256, 3, activation='relu')(x)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

@app.route('/predict', methods=['POST'])
def predict():

    with open('resources/tokenizer.json') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    model = get_model(tokenizer)
    model.load_weights("resources/lstm_toxic_1.h5")

    if request.method == 'POST':
        comment = str(request.form['comment'])
        input = tokenize(list(comment), tokenizer)
        prediction = model.predict(input)
        labels = ['toxic', 'severe_toxic', 'obscene',
          'threat', 'insult', 'identity_hate']
        category = str(labels[np.argmax(prediction[0])])

    return render_template(
        'result.html',
        prediction=prediction[0], 
        labels=labels, 
        category=category
    )

if __name__ == '__main__':
    app.run(debug=True)