PREDICT CLOSED QUESTIONS ON STACK OVERFLOW
============

## Project dependencies


This project aims to develop a machine learning model using Bidirectional Encoder Representations from Transformers (BERT) to predict whether a question posted on Stack Overflow will be closed. This can assist moderators in identifying potentially low-quality questions and improve the overall platform's efficiency.

Requirements
--------

* Python 3.6+
* Necessary libraries (install using `pip install <library_name>`)
  *   `transformer`
  *   `torch` (GPU-accelerated training highly recommended)
  *   `pandas`
  *   `numpy`
  *   `sklearn` (for preprocessing)

DATA
---------------

* The code assumes you have access to a Stack Overflow dataset containing questions (textual content), labels indicating whether they were *open*, *not a real question*, *not constructive*, *too localized* or *off topic*, and potentially additional features like tags, timestamps, or user information.
* Download or prepare your dataset accordingly.

## Model Architecture
* This code utilizes a fine-tuned BERT model for text classification. BERT effectively captures contextual relationships within text, making it well-suited for this task.

**This is the architecture on which the model is trained**

  ![download (1)](https://github.com/user-attachments/assets/f2abe795-509f-4e11-9e9e-b8ecef660b5a)

## Code Structure

* `tokenizer.py`: This Python file (tokenizer.py) is designed to implement a tokenizer function that breaks down text strings into smaller units called tokens. These tokens can be words, punctuation marks, characters, or n-grams (sequences of n words) depending on the specific tokenization method employed.
* `preprocessing.py`: This is a Python file (preprocessing.py) which is designed to implement on to extract all the important columns in the given dataset and to concat to form a single column containing all the important information and the tokenization is applied on this column and this is send as input in input layer of BERT
