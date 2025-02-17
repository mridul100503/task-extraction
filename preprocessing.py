#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy nltk')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[2]:


get_ipython().system('python.exe -m pip install --upgrade pip')


# In[3]:


import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Ensure Required NLTK Resources are Available ---
required_resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet'
}

for resource, path in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# --- Global Variables to Avoid Repeated Work ---
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- Clean Text ---
def clean_text(text):
    """
    Clean the input text by converting to lowercase, removing punctuation,
    and extra spaces.
    """
    if isinstance(text, dict):
        text = text.get("text", "")
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- Sentence Tokenization ---
def tokenize_sentences(text):
    """
    Tokenize the input text into sentences.
    """
    return sent_tokenize(text)

# --- Word Tokenization ---
def tokenize_words(sentence):
    """
    Tokenize a sentence into words using NLTK's word_tokenize.
    """
    return word_tokenize(sentence)

# --- Stopword Removal ---
def remove_stopwords(tokens):
    """
    Remove English stopwords from a list of tokens.
    """
    return [word for word in tokens if word.lower() not in STOPWORDS]

# --- POS Tagging ---
def pos_tagging(tokens):
    """
    Perform Part-of-Speech (POS) tagging on a list of tokens.
    """
    return nltk.pos_tag(tokens)

# --- Preprocess Text ---
def preprocess_text(text):
    """
    Preprocess the input text by performing the following steps:
      1. Clean the text (lowercase, remove punctuation, extra spaces)
      2. Tokenize the text into words
      3. Remove stopwords
      4. Perform POS tagging on the tokens

    Returns:
        dict: A dictionary containing:
              - 'cleaned_text': The cleaned text.
              - 'tokens': List of tokens.
              - 'filtered_tokens': Tokens after stopword removal.
              - 'pos_tags': List of tuples with POS tags.
    """
    cleaned = clean_text(text)
    tokens = tokenize_words(cleaned)
    filtered_tokens = remove_stopwords(tokens)
    pos_tags = pos_tagging(tokens)
    
    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "pos_tags": pos_tags
    }




# In[ ]:




