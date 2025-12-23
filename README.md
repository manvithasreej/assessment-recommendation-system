# Assessment Recommendation System

A lightweight machine learning–based recommendation system that suggests the most relevant assessments based on a user’s role or skill requirements.

The project focuses on clean architecture, explainable logic, and real-world engineering practices, rather than complex black-box AI.

# Problem Statement

Selecting the right assessments is often manual or opaque. This system automates the process by matching user intent with assessment descriptions in a clear and scalable way.

# Solution Approach

Convert assessment descriptions into vector embeddings

Convert user queries into embeddings

Compute similarity using cosine similarity

Rank and return the most relevant assessments

This approach is fast, explainable, and commonly used in production systems.


# Tech Stack

Python

Sentence embeddings

Cosine similarity

Modular ML design

# How to Run
pip install -r requirements.txt 

python app.py
