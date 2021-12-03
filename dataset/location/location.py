#!/usr/bin/python
# -*- coding:utf-8 -*-
import pickle
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk.sentiment.util as util
import nltk.sentiment.sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from scipy.stats import norm


mean = None
sigma = None
nlp = spacy.load('en_core_web_sm')


def get_location_words(text, pos=False):
    locations = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            locations.append(ent.text)
        elif ent.label_ == 'LOC':
            locations.append(ent.text)
    return locations


def judge_location(text):
    global mean, sigma
    if mean is None or sigma is None:
        path = __file__.replace('location.py', 'stats.data')
        with open(path, 'rb') as fin:
            mean, sigma = pickle.load(fin)
    keywords = get_location_words(text)
    val = (len(keywords) - mean) / sigma
    return norm.cdf(val)


# if __name__ == '__main__':
#     print(__file__)
#     for text in examples:
#         print(text)
#         print(get_emotional_words(text, pos=True))
#         print(judge_emotional(text))
