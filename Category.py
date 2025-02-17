#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import contextlib
import logging

# Suppress verbose logs from gensim, nltk, and spacy
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("nltk").setLevel(logging.ERROR)
logging.getLogger("spacy").setLevel(logging.ERROR)

# -----------------------------
# 1. SETUP & NLTK RESOURCE DOWNLOAD
# -----------------------------
import nltk
import spacy    # Added import spacy
import numpy as np  # Added import numpy

# Redirect output to suppress "Requirement already satisfied" messages
with open(os.devnull, "w") as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("wordnet", quiet=True)


# In[5]:


from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text
from taskidentification import extract_tasks


# In[10]:


# Predefined categories
predefined_categories = [
    "Deep Learning & Image Processing",
    "Web Development & JavaScript",
    "Chatbots & Transformers",
    "NLP & Data Preprocessing",
    "Database Optimization",
    "Hybrid Chatbot & Web",
    "Computer Vision & Accident Detection"
]


def prepare_category_training_data(categories):
    return [preprocess_text(cat)["tokens"] for cat in categories]

training_sentences = prepare_category_training_data(predefined_categories)


# Train Word2Vec model once
w2v_model = Word2Vec(
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)
w2v_model.build_vocab(training_sentences)
w2v_model.train(training_sentences, total_examples=w2v_model.corpus_count, epochs=20)


# In[11]:


def get_phrase_vector(phrase, model):
    """
    Get average vector for a phrase based on its tokens.
    """
    if isinstance(phrase, dict):
        words = phrase.get("tokens", phrase.get("task", ""))
        if isinstance(words, str):
            words = words.split()
    elif isinstance(phrase, list):
        words = phrase
    else:
        words = phrase.split()
        
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else None

# Precompute category vectors
precomputed_category_vectors = {}
for category in predefined_categories:
    cat_processed = preprocess_text(category)
    cat_vector = get_phrase_vector(cat_processed, w2v_model)
    precomputed_category_vectors[category] = cat_vector

def find_best_category_phrase(task, categories, model, threshold=0.2):
    """
    Find the best matching category for a given task.
    """
    task_text = task.get("task", "") if isinstance(task, dict) else task
    task_processed = preprocess_text(task_text)
    task_vector = get_phrase_vector(task_processed, model)
    if task_vector is None:
        return None
    
    best_category = None
    max_similarity = -1
    for category in categories:
        cat_vector = precomputed_category_vectors.get(category)
        if cat_vector is not None:
            sim = cosine_similarity(task_vector.reshape(1, -1), cat_vector.reshape(1, -1))[0][0]
            if sim > max_similarity:
                max_similarity = sim
                best_category = category
    return best_category if max_similarity >= threshold else None

# Load spaCy model once (suppressing extra logs)
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])

def extract_fallback_category(task_input):
    """
    Use spaCy to extract a fallback category based on the task's verb structure.
    """
    task_text = task_input.get("task", "") if isinstance(task_input, dict) else task_input
    doc = nlp(task_text)
    # Try subordinate verb
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("xcomp", "advcl", "conj"):
            dobj = next((child for child in token.children if child.dep_ in ("dobj", "obj")), None)
            if dobj:
                return f"{dobj.lemma_} {token.lemma_} task"
    # Try ROOT verb
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            dobj = next((child for child in token.children if child.dep_ in ("dobj", "obj")), None)
            return f"{dobj.lemma_} {token.lemma_} task" if dobj else f"{token.lemma_} task"
    return "other task"


# In[12]:


def task_pipeline(user_input):
    """
    Main pipeline: Extract tasks from input text, match them to predefined categories,
    and apply fallback categorization if needed.
    """
    tasks = extract_tasks(user_input)
    print("Extracted Tasks:")
    for t in tasks:
        print(" -", t)
    
    final_mapping = {}
    for task in tasks:
        task_text = task.get("task", "") if isinstance(task, dict) else task
        match = find_best_category_phrase(task, predefined_categories, w2v_model, threshold=0.2)
        final_mapping[task_text] = match if match else extract_fallback_category(task)
    return final_mapping


# In[ ]:




