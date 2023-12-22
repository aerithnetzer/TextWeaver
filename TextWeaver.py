# This module will handle textual applications of qualitative data analysis

# Import the necessary libraries
import nltk
import os
from colorama import Fore, Back, Style
import time
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
        self.text = text
        self.tokens = word_tokenize(self.text)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.codes = {}  # instantiate Codes class 

    def assign_codes(self, key, value):
        """
        Assigns codes to the text.

        Parameters:
        - codes (dict): The dictionary of codes.
        """

        if key in self.codes:
            self.codes[key].append(value)
        else:
            self.codes[key] = [value]
    
    def find_codes(self, key):
        """
        Takes as input the text and the codes and returns the indices where the key is found in the text.

        Returns:
        - list: list of indices where the key is found in the text.
        """

        text = self.text
        codes = self.codes

        for value in codes[key]:
            return [i for i in range(len(text)) if text.startswith(value, i)]

    
    from colorama import Fore, Back, Style

    def find_themes(self, theme):
        """
        Takes as input the text and the keys of dictionary codes and returns the indices where the values of each key is found in the text.

        Returns:
        - list: list of indices where the key is found in the text.
        """

        text = self.text
        codes = self.codes
        indices = [(i, i + len(value) - 1) for value in codes[theme] for i in range(len(text)) if text.startswith(value, i)]

        # Reset the color to its original state
        colored_text = Style.RESET_ALL
        last_index = 0
        for start, end in indices:
            # Add the non-highlighted part
            colored_text += text[last_index:start]
            # Add the highlighted part
            colored_text += Fore.GREEN + text[start:end+1] + Style.RESET_ALL
            last_index = end + 1
        # Add the remaining non-highlighted part
        colored_text += text[last_index:]

        print(colored_text)
    
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
            - dict: The sentiment scores of the text.
            """
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(self.text)
            return sentiment
class Garment:
    """
    A class representing a collection of fabrics.

    Attributes:
        directory (str): The directory path where the fabrics are located.
        corpus (dict): A dictionary containing the loaded fabrics.

    Methods:
        load_corpus(): Loads the fabrics from the directory.
        get_sentences(): Returns the sentences from each fabric in the corpus.
        get_lemmas(): Returns the lemmas from each fabric in the corpus.
        get_stems(): Returns the stems from each fabric in the corpus.
        remove_stopwords(): Removes stopwords from each fabric in the corpus.
        get_sentiment(): Returns the sentiment of each fabric in the corpus.
    """

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