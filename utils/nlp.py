import nltk
import string
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import RSLPStemmer

def to_lower(text:str)-> str:
    """
    Get input sentence and returns all words as lowercase
    """
    return text.lower()

def remove_symbols(text:str)-> str:
    """
    Get the input text and replaces all symbols from english language for spaces.
    This procedure aims to later remove the extra spaces when tokenizing words.
    """
    # text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~])', ' ', text)
    return text

def word_tokenize(text:str)->list:
    """
    Break the input string text into word tokens, removing spaces between them.
    """
    words = nltk.word_tokenize(text)
    return words

def remove_stopwords(words:list)->list:
    """
    Remove stopwords from english text from a list of words.
    """
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def remove_numbers(words:list)->list:
    """
    Remove tokens that are only composed by numbers, since they present almost none semantic information.
    """
    words = [w for w in words if not w.isdecimal()]
    return words

def apply_stemming(words:list)->list:
    """
    Apply a stemmer technique in the tokenized words.
    TODO: Implement other stemming techniques as option.
    """
    stemmer = RSLPStemmer()
    stemmed = [stemmer.stem(word) for word in words]
    return stemmed

def remove_single_letters(words:list)->list:
    """
    Similar to numbers, remove words with only one letter.
    """
    words = [w for w in words if len(w) > 1]
    return words

def process_sentence(text:str,
                     process_symbols:bool = True,
                     process_stopwords:bool = True,
                     process_numbers:bool = True,
                     process_single_letters:bool = True)->list:
    """
    Get a raw sentence and applies all preprocessing stages.
    """
    text = to_lower(text)
    
    if process_symbols:
        text = remove_symbols(text)
    
    words = word_tokenize(text)
    # print("without symbols", len(words))
    
    if process_stopwords:
        words = remove_stopwords(words)
    # print("without stopwords", len(words))
    
    if process_numbers:
        words = remove_numbers(words)
    # print("without numbers", len(words))
    
    if process_single_letters:
        words = remove_single_letters(words)
    
    words = apply_stemming(words)
    
    return words

def process_dataset(dataset: pd.core.frame.DataFrame,
                    process_symbols:bool = True,
                    process_stopwords:bool = True,
                    process_numbers:bool = True,
                    process_single_letters:bool = True,
                    debug:bool = True) -> list:
    """
    Apply the sentence preprocessing over a complete pd.DataFrame dataset.
    """
    processed = []
    total = len(dataset)
    for index, row in dataset.iterrows():
        if debug:
            print(f"Processing {index+1}/{total}", end="\r")
        processed.append(process_sentence(row["text"]),
                         process_symbols,
                         process_stopwords,
                         process_numbers,
                         process_single_letters)
    
    return processed

def create_word_corpus(dataset:list,
                       min_percentage:float = 0.0,
                       max_percentage:float = 1.0)->tuple:
    """
    Creates a word_corpus selection based on a processed dataset.
    It also returns the term_count for the corpus, allowing for later document encoding
    """
    raw_word_corpus = [w for item in dataset for w in item]
    raw_word_corpus = list(set(word_corpus))

    term_count = {w:0 for w in raw_word_corpus}
    for text in dataset:
        for word in text:
            term_count[word] += 1
    sorted_term_count = {k: v for k, v in sorted(term_count.items(), key=lambda item: item[1], reverse=True)}
    
    term_df = pd.DataFrame(list(sorted_term_count.items()), columns=["word", "count"])
    term_df["average_count"] = term_df["count"]/len(dataset)
    
    word_corpus_term_count = term_df[term_df["average_count"].between(min_percentage, max_percentage)]
    word_corpus = word_corpus_term_count["word"].to_list()

    return word_corpus, word_corpus_term_count