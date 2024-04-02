# NYU CS-469 Natural Language Processing Final Project
This repository contains the code and resources for our final project in NYU's CS-469 Natural Language Processing class.

## Project Description
Our project aims to develop a system that automatically predicts and vocalizes unvocalized Arabic text using various NLP methods.

Arabic text is often written without vowel markings, which can be challenging for non-native speakers. Our system will address this challenge by predicting the appropriate vowels based on the context of the surrounding letters.

## Here's an overview of the project:
**Problem:** Automatically predict and vocalize unvocalized Arabic text. 

**Techniques:** We will explore three NLP techniques: 

* N-gram model with Viterbi algorithm

* RNN-based approach

* Transformer-based architecture 

**Data:** 
Pre-vocalized text from the Quran and Hadith  

Children's novels and beginner Arabic books  

Potentially the Tashkeel dataset 

**Evaluation:** We will evaluate our system's accuracy on a held-out test set, aiming for above 90% accuracy on the development set. 

## Getting Started
This repository includes the following:

Code for the NLP models 

Data pre-processing scripts 

Evaluation scripts 

Documentation 

## Prerequisites:

Python 3.6 or later
Necessary libraries (to be specified in a requirements.txt file)

## Instructions:

Clone this repository. 

Install the required libraries using 

```bash
pip install -r requirements.txt
```

Refer to the documentation (to be added) for further instructions on running the code and evaluating the models. 

## Collaboration Plan
The project is divided among team members as follows: 


Ben: Administration of testing, measuring success, and data collection 

Cheng: Linguistics, data collection, and software review 

Reyhan: Software development and English writing 

Usaid: Programming and development of algorithms 

We will develop the model concurrently with data collection and leverage transfer learning for training.

## Future Work 
Explore additional NLP techniques for improvement. 

Investigate methods to handle unseen words or rare contexts. 
