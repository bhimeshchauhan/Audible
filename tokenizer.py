import re

def sentence_split(text):
    return re.split(r'(?<=[.!?]) +', text)

def word_tokenize(sentence):
    return re.findall(r'\\b\\w+\\b', sentence)
