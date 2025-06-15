# Spam-Detection-using-classical-NLP

A Simple, fast, and effective mail spam detection system built using classical Natural Language Processing techniques and Machine Learning.

## Project Summary

This project uses the SMS mail Collection collection dataset to classify messages as spam or ham (not spam).
We applied basic text processing, CountVectorization and train a Logistic Regression and Naive Bayes classifier for prediction.

Despite exploring n-grams and model variation, the unigram based  Vectorization + Naive Bayes setup gave the best balance of speed and accuracy.

## Result
- Accuracy: 97 % on test set
- Precision (Spam): High
- Model Size: Less than 2 MB
- Inference time: < 50ms per message on CPU

## Tech Stack
- Python
- Scikit-Learn
- NLTK
- Pandas
- Seaborn / Matplotlib
- Jupyter Notebook

## Model Pipeline
- Load & Explore Data
- Clean Text (Basic NLP Text Preprocessing)
- Vectorize with CountVectorizer
- Train Log Reg Model and MultinomialNL model
- Evaluate with accrucy, confusion matrix, and classification report
- Save model using joblib for inference

## Why simple?
- Despite tyring bigrams and trigrams, they added complexity with little to no performance gain.
- Classical models like Naive Bayes, paired with clean text and good features, still hold for strong, fast, interpretable NLP tasks

## Future Improvements
- Try word embeddings (Word2Vec, GloVe)
- Deploy using Streamlist or FastAPI
- Experiment with LLMs-based spam detection

## Author
- Suel Ahmed
- Actively building a portfolio of real-world NLP Projects

**Part of my 20-Day MLP Spring Day#2**