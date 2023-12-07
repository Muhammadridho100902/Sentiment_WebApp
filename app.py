import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

loaded_model = load_model('my_model.h5')
with open('my_tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file) 
    
def predictedSentences(text):
    
    def remove_url(text):
        # http://google.com adalah bla bla bla
        # https://youtube.com
        return re.sub(r'http\S+', '', text)

    def remove_number(text):
        # 123 saya blablabla
        return re.sub(r'\d+', '', text)

    def remove_stopwords(text):
        words = text.split()
        new_text = ""
        for word in words:
            if word not in stopwords.words('english'):
                new_text += word
                new_text += " "

        return new_text

    def lemmatized_text(text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        new_text = ""
        for word in words:
            lemmatized_word = lemmatizer.lemmatize(word)
            new_text += lemmatized_word
            new_text += " "
        return new_text

    def stemmed_text(text):
        stemmer = PorterStemmer()
        words = text.split()
        new_text = ""
        for word in words:
            stemmed_word = stemmer.stem(word)
            new_text += stemmed_word
            new_text += " "
        return new_text

    def preprocessing(text):
        # lowercase -> Running, running, RUNNING, RunNinG
        text = text.lower()

        # remove url
        text = remove_url(text)

        # remove number
        text = remove_number(text)

        # remove stopwords
        text = remove_stopwords(text)

        # lemmatization
        text = lemmatized_text(text)

        # stemming
        text = stemmed_text(text)

        return text

    preprocessed_text = preprocessing(text)
    return preprocessed_text

def predictSentiment():
        model = load_model('my_model.h5')

        with open('my_tokenizer.pickle', 'rb') as file:
            tokenizer = pickle.load(file)
            
        st.title('Sentiment Prediction')
        text = st.text_input('Input Sentences')
        
        if text == "":
            st.text("")
        else:
            preprocessed_text = predictedSentences(text)
            sequence = tokenizer.texts_to_sequences([preprocessed_text])
            padded_sequences = pad_sequences(sequence, padding = 'pre', truncating= 'pre', maxlen = 83)
            result = model.predict(padded_sequences)
            output_array = np.array(result)
            predicted_category_index = np.argmax(output_array)
            
            # Mendefinisikan kategori
            categories = ["Positive", "Negative", "Neutral"]
            
            # Mendapatkan kategori berdasarkan indeks
            predicted_category = categories[predicted_category_index]
            st.text(predicted_category)

if __name__ == '__main__':
    predictSentiment()