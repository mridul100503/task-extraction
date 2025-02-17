# Extracting and Categorizing Tasks from Unstructured Text

## Overview
This project implements an NLP pipeline designed to extract actionable tasks from unstructured text and categorize them. The system identifies sentences containing actionable instructions and extracts key details such as the responsible person and deadlines. It then organizes the tasks into meaningful groups using a heuristic-based approach, with options for additional topic modeling.

## Objectives
- **Task Extraction:** Identify actionable sentences using imperative detection and task indicator phrases.
- **Entity Extraction:** Extract the responsible person from task sentences when available.
- **Deadline Extraction:** Capture deadline information (e.g., "by 5 pm today", "by next Monday") using robust regular expressions.
- **Task Categorization:** Classify the extracted tasks into predefined categories (e.g., Shopping, Cleaning, Communication, Review, Work, Errand) using keyword matching and, optionally, LDA for dynamic topic clustering.

## Components

### Preprocessing
- **Text Cleaning:** Remove punctuation and unwanted characters while preserving the original sentence for extraction.
- **Tokenization:** Split the text into sentences and words.
- **POS Tagging:** Use part-of-speech tagging to help identify imperative sentences and other key elements.

### Task Identification
- **Imperative Detection:** Determine if a sentence is a command by checking if it begins with a base-form verb.
- **Task Indicators:** Look for phrases such as "has to", "should", "must", "needs to", and "ought to" to signal actionable tasks.
- **Deadline Extraction:** Apply comprehensive regex patterns to extract deadlines from task sentences.
- **Person Extraction:** Use regex to capture the responsible person's name when it appears immediately before task indicators.

### Task Categorization
- **Keyword-Based Categorization:** Assign tasks to categories based on predefined keyword lists.
- **Optional LDA Topic Modeling:** Optionally, use LDA to cluster tasks dynamically for additional insights.

## Output
The pipeline produces a structured list of tasks that includes:
- **Task Sentence:** The original sentence describing the task.
- **Person:** The extracted responsible individual (if detected).
- **Deadline:** The extracted deadline (if detected).
- **Category:** The assigned category based on keyword matching or topic modeling.
