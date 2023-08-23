import tensorflow as tf
import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

vocal_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
epochs = 20

def prepare_data():
    articles = []
    labels = []

    with open('bbc-text.csv','r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in stop_words:
                token = ' '+ word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            articles.append(article)
    
    print(f'[INFO] lenght of arcticle {len(articles)}')
    print(f'[INFO] lenght of labels {len(labels)}')
    return articles, labels

def train_test_split(articles, labels):
    train_size = int(len(articles)*0.8)

    train_articles = articles[ : train_size]
    train_labels = labels[: train_size]

    val_articles = articles[train_size: ]
    val_labels = labels[train_size: ]

    print(f'[INFO] lenght of train arcticle {len(train_articles)}') 
    print(f'[INFO] lenght of train label {len(train_labels)}') 
    print(f'[INFO] lenght of validation arcticle {len(val_articles)}') 
    print(f'[INFO] lenght of validation label {len(val_labels)}')

    return train_articles , train_labels , val_articles , val_labels     

def preprocessing(train_articles, val_articles):
    tokenizer = Tokenizer(num_words=vocal_size, oov_token= oov_tok)
    tokenizer.fit_on_texts(train_articles)
    work_index = tokenizer.word_index

    #generating the sequence
    train_sequences = tokenizer.texts_to_sequences(train_articles)
    val_sequences = tokenizer.texts_to_sequences(val_articles)

    # padding the data
    pad_train_article = pad_sequences(train_sequences, maxlen = max_length, padding=padding_type , truncating= trunc_type)

    pad_val_article = pad_sequences(val_sequences, maxlen = max_length, padding=padding_type , truncating= trunc_type)

    return pad_train_article, pad_val_article

def tokenize_label(labels, train_labels, val_labels):
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    val_label_seq = np.array(label_tokenizer.texts_to_sequences(val_labels))
    
    return train_label_seq, val_label_seq

def train_model(pad_train_article, pad_val_article, train_label_seq, val_label_seq):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocal_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(pad_train_article, train_label_seq, epochs=epochs, validation_data=(pad_val_article, val_label_seq ), verbose = 2 )
 

if __name__ == '__main__':
    articles, labels = prepare_data()
    train_articles, train_labels, val_articles, val_labels =  train_test_split(articles, labels)
    pad_train_article,pad_val_article = preprocessing(train_articles , val_articles)
    train_label_seq, val_label_seq =tokenize_label(labels, train_labels, val_labels)
    train_model(pad_train_article, pad_val_article, train_label_seq, val_label_seq )



