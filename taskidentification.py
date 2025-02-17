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
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from preprocessing import tokenize_sentences, clean_text
from datetime import datetime, timedelta

# --- Precompile Regex Patterns ---
TASK_INDICATORS_PATTERNS = [
    re.compile(r"\bhas to\b", re.IGNORECASE),
    re.compile(r"\bneed to\b", re.IGNORECASE),
    re.compile(r"\bneeds to\b", re.IGNORECASE),
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bought to\b", re.IGNORECASE)
]

PERSON_REGEX = re.compile(
    r'((?:[A-Za-z][a-z]*\s*)+)(?=\s+(has to|need to|needs to|should|must|ought to))',
    re.IGNORECASE
)

DEADLINE_REGEX = re.compile(
    r'\b(?:by|before|at)\s+'
    r'((?:(?:\d{1,2}\s*(?:am|pm))|noon|midday|midnight)(?:\s+\w+)?|'
    r'tomorrow|today|tonight|next\s+\w+day|\w+day|end of day|early morning|late night)'
    r'(?=\b|[\s\.,]|$)',
    re.IGNORECASE
)

# Compile pattern for splitting compound tasks
SPLIT_PATTERN = re.compile(r',\s*| and ')

# --- Helper Functions ---

def tokenize_words(sentence):
    """Tokenize a sentence into words."""
    return word_tokenize(sentence)

def pos_tagging(tokens):
    """Perform POS tagging on a list of tokens."""
    return pos_tag(tokens)

def is_imperative(sentence):
    """
    Check if a sentence is likely an imperative sentence.
    
    Heuristic:
    - Remove polite words like 'please' and 'kindly'
    - If the first meaningful word is a base form verb (VB), it's likely imperative.
    """
    polite_words = {"please", "kindly"}
    tokens = tokenize_words(sentence)
    if not tokens:
        return False

    filtered_tokens = [word.lower() for word in tokens if word.lower() not in polite_words]
    if not filtered_tokens:
        return False

    tagged = pos_tagging(filtered_tokens)
    return tagged[0][1] == 'VB'

def contains_task_indicators(sentence):
    """
    Check if the sentence contains phrases that indicate a task.
    Uses a set of precompiled indicator patterns.
    """
    sentence_lower = sentence.lower()
    return any(pattern.search(sentence_lower) for pattern in TASK_INDICATORS_PATTERNS)

def extract_person(sentence):
    """
    Extract the person's name from the sentence if it precedes a task indicator phrase.
    """
    match = PERSON_REGEX.search(sentence)
    return match.group(1).strip() if match else None

def extract_deadline(sentence):
    """
    Attempt to extract deadline information from the sentence.
    Heuristic: Look for phrases starting with 'by', 'before', or 'at' followed by time expressions or day indicators.
    """
    match = DEADLINE_REGEX.search(sentence)
    return match.group(1) if match else None

def remove_ordinal_suffix(date_str):
    """Remove ordinal suffixes ('st', 'nd', 'rd', 'th') from date strings."""
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

def convert_to_datetime(deadline):
    """
    Convert an extracted deadline string into a datetime object, if possible.
    
    - For relative dates like "tomorrow" or "today", uses the current date.
    - For time-only strings like "5 pm", attaches today's date.
    - For absolute dates like "March 10th", assumes the current year.
    """
    today = datetime.today()
    deadline_str = deadline.strip()
    deadline_lower = deadline_str.lower()

    if "tomorrow" in deadline_lower:
        return today + timedelta(days=1)
    elif "today" in deadline_lower or "tonight" in deadline_lower:
        return today

    time_match = re.match(r'(\d{1,2})\s*(am|pm)', deadline_str, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(2).lower()
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return today.replace(hour=hour, minute=0, second=0, microsecond=0)

    deadline_clean = remove_ordinal_suffix(deadline_str).strip()
    try:
        parsed_date = datetime.strptime(deadline_clean, "%B %d")
        return parsed_date.replace(year=today.year)
    except ValueError:
        return None

def split_compound_tasks(sentence):
    """
    Splits a compound sentence into individual task clauses using commas and 'and' as delimiters.
    If the first clause contains a subject (e.g., "I need to"), then that subject is prepended to subsequent
    clauses that appear to be missing it.
    
    Example:
        "I need to fix my website, improve database performance, and develop a new AI chatbot."
    becomes:
        [
            "I need to fix my website",
            "I need to improve database performance",
            "I need to develop a new AI chatbot"
        ]
    """
    splits = SPLIT_PATTERN.split(sentence)
    splits = [s.strip() for s in splits if s.strip()]
    if not splits:
        return [sentence]
    
    first_clause = splits[0]
    subject_phrase = ""
    tokens = first_clause.split()
    if tokens:
        if tokens[0].lower() in ["i", "we", "you"]:
            if len(tokens) > 1 and tokens[1].lower() in ["need", "should", "must", "have", "want"]:
                subject_phrase = " ".join(tokens[:2])
            else:
                subject_phrase = tokens[0]
    
    if subject_phrase:
        for i in range(1, len(splits)):
            clause_tokens = splits[i].split()
            if clause_tokens and clause_tokens[0].lower() not in subject_phrase.lower():
                splits[i] = subject_phrase + " " + splits[i]
    
    return splits

def extract_tasks(raw_text):
    """
    Extract tasks from raw text by identifying imperative sentences, 
    task indicator phrases, and deadlines. Splits compound tasks into separate entries.
    
    Returns:
        A list of dictionaries with keys: 'task', 'person', 'deadline'
    """
    sentences = tokenize_sentences(raw_text)
    tasks = []
    
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        deadline = extract_deadline(cleaned_sentence)  # Extract deadline once per sentence
        
        if is_imperative(sentence) or contains_task_indicators(cleaned_sentence) or deadline:
            sub_tasks = split_compound_tasks(sentence)
            for sub in sub_tasks:
                person = extract_person(sub) or "Unassigned"
                tasks.append({
                    'task': sub,
                    'person': person,
                    'deadline': deadline
                })
        else:
            tasks.append('This is not Task')
    
    return tasks
