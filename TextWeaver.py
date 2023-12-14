# This module will handle textual applications of qualitative data analysis
# It does the following:
# Sentiment analysis

# Import the necessary libraries
import os
import re
import sys
import nltk

# Import the necessary modules
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk import ne_chunk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Create a class for Fabric, which will represent the text

class Fabric:
    def __init__(self, text):
        """
        Initializes a Fabric object.

        Parameters:
        - text (str): The input text.

        Attributes:
        - text (str): The input text.
        - tokens (list): The list of tokens extracted from the text.
        - lemmatizer (WordNetLemmatizer): The WordNet lemmatizer object.
        - stemmer (PorterStemmer): The Porter stemmer object.
        - stop_words (set): The set of stop words in English.
        """
        self.text = text
        self.tokens = word_tokenize(self.text)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def get_pos(self):
        """
        Returns the part-of-speech tags for the tokens.

        Returns:
        - list: The list of part-of-speech tags.
        """
        return pos_tag(self.tokens)

    def get_named_entities(self):
        """
        Returns the named entities in the text.

        Returns:
        - nltk.tree.Tree: The named entities in the text.
        """
        return ne_chunk(self.get_pos())

    def get_sentences(self):
        """
        Returns the sentences in the text.

        Returns:
        - list: The list of sentences.
        """
        return sent_tokenize(self.text)

    def get_lemmas(self):
        """
        Returns the lemmas of the tokens.

        Returns:
        - list: The list of lemmas.
        """
        return [self.lemmatizer.lemmatize(token) for token in self.tokens]

    def get_stems(self):
        """
        Returns the stems of the tokens.

        Returns:
        - list: The list of stems.
        """
        return [self.stemmer.stem(token) for token in self.tokens]

    def remove_stopwords(self):
        """
        Removes the stop words from the tokens.

        Returns:
        - list: The list of tokens without stop words.
        """
        return [token for token in self.tokens if not token in self.stop_words]
    
    def get_sentiment(self):
        """
        Returns the sentiment of the text.

        Returns:
        - int: The sentiment of the text.
        """

    def get_sentiment(self):
            """
            Returns the sentiment of the text.

            Returns:
            - dict: The sentiment scores of the text.
            """
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(self.text)
            return sentiment

class Garment:

    def __init__(self, directory):
        self.directory = directory
        self.corpus = self.load_corpus()

    def load_corpus(self):
        corpus = {}
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                with open(os.path.join(self.directory, filename), 'r') as f:
                    corpus[filename] = Fabric(f.read())
        return corpus

    def get_sentences(self):
        return {filename: fabric.get_sentences() for filename, fabric in self.corpus.items()}

    def get_lemmas(self):
        return {filename: fabric.get_lemmas() for filename, fabric in self.corpus.items()}

    def get_stems(self):
        return {filename: fabric.get_stems() for filename, fabric in self.corpus.items()}

    def remove_stopwords(self):
        return {filename: fabric.remove_stopwords() for filename, fabric in self.corpus.items()}
    
    def get_sentiment(self):
        return {filename: fabric.get_sentiment() for filename, fabric in self.corpus.items()}