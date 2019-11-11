import pickle
import os
from nltk.tokenize import sent_tokenize

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
classifier = pickle.load(open('models/classifier.sav', 'rb'))

def analyze_sentiment(text):
    text_vector = vectorizer.transform([text])
    result = classifier.predict(text_vector)
    return result[0]

def get_paragraph_sentiment(paragraph):
    results = []
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        res = {}
        res['sentence'] = sentence
        res['sentiment'] = analyze_sentiment(sentence)
        results.append(res)
    return (paragraph, results)