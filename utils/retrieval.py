import pandas as pd
import numpy as np
import math
from abc import ABC, abstractmethod
from sklearn import metrics
from sentence_transformers import SentenceTransformer


class VectorRetrievalModel(ABC):
 
    def __init__(self, word_corpus: pd.DataFrame):
        self.word_corpus = word_corpus.word.to_list()
        # self.word_corpus.sort()
        self.dataset = None
        self.metric = None
        super().__init__()
        
    @abstractmethod
    def convert_dataset(self, dataset:list):
        pass
    
    @abstractmethod
    def convert_item(self, item:list) -> list:
        pass

    def get_dataset(self) -> np.ndarray:
        return self.dataset

    def get_ranked_lists(self) -> np.ndarray:
        return self.ranked_lists
    
    def compute_ranked_lists(self, metric:str = None, test_filter=None) -> np.ndarray:
        if self.dataset is None:
            raise Exception("No dataset has been defined, please create a new one with convert_dataset function.")
        
        if self.metric is None:
            raise Exception("No metric defined for the model.")
        
        if metric is None:
            metric = self.metric
            
        target_ranked_lists = self.dataset if test_filter is None else self.dataset[test_filter]
            
        
        ranked_lists = []

        if metric == "cosine":
            distances = metrics.pairwise.cosine_distances(target_ranked_lists, self.dataset)
        else:
            distances = metrics.pairwise.euclidean_distances(target_ranked_lists, self.dataset)
        
        for item in distances:
            rank_map = np.argsort(item)
            ranked_lists.append(rank_map)
        self.ranked_lists = np.asarray(ranked_lists)


class BinaryModel(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = 'cosine'
        
    def convert_dataset(self, dataset:list):
        binary_dataset = []
        for item in dataset:
            binary_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(binary_dataset)
    
    def convert_item(self, item:list)-> list:
        binary_item = []
        for word in self.word_corpus:
            value = 1 if word in item else 0
            binary_item.append(value)
        return binary_item


class BagOfWordsModel(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = "cosine"
        
    def convert_dataset(self, dataset:list):
        bow_dataset = []
        for item in dataset:
            bow_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(bow_dataset)
    
    def convert_item(self, item:list)-> list:
        bow_item = []
        for word in self.word_corpus:
            bow_item.append(item.count(word))
        return bow_item


class TfIdfModel(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = "cosine"
    
    def compute_idf(self, dataset:list):
        word_idf = []
        dataset_size = len(dataset)
        for index, word in enumerate(self.word_corpus):
            word_idf.append(0)
            for item in dataset:
                if word in item:
                    word_idf[index] += 1
            if word_idf[index] == 0:
                continue
            word_idf[index] = math.log(dataset_size/word_idf[index], 2)
        self.word_idf = np.asarray(word_idf)
        
    def convert_dataset(self, dataset:list):
        # Calcular o idf para cada palavra.
        self.compute_idf(dataset)
        
        tf_idf_dataset = []
        for item in dataset:
            tf_idf_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(tf_idf_dataset)
    
    def compute_tf(self, word:str, item:list):
        word_count = item.count(word)
        if word_count == 0:
            return 0
        else:
            return 1 + math.log(word_count, 2)
    
    def convert_item(self, item:list)-> list:
        tf_idf_item = []
        for index, word in enumerate(self.word_corpus):
            tf = self.compute_tf(word, item)
            tf_idf = tf * self.word_idf[index]
            tf_idf_item.append(tf_idf)
        return tf_idf_item


class BM1Model(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = "cosine"
    
    def compute_idf(self, dataset:list):
        word_idf = []
        dataset_size = len(dataset)
        for index, word in enumerate(self.word_corpus):
            word_idf.append(0)
            for item in dataset:
                if word in item:
                    word_idf[index] += 1
            word_idf[index] = math.log((dataset_size - word_idf[index] + 0.5) / (word_idf[index] + 0.5), 2)
        self.word_idf = np.asarray(word_idf)
        
    def convert_dataset(self, dataset:list):
        # Calcular o idf para cada palavra.
        self.compute_idf(dataset)
        
        tf_idf_dataset = []
        for item in dataset:
            tf_idf_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(tf_idf_dataset)
    
    def compute_tf(self, word:str, item:list):
        word_count = item.count(word)
        if word_count == 0:
            return 0
        else:
            return 1 + math.log(word_count, 2)
    
    def convert_item(self, item:list)-> list:
        tf_idf_item = []
        for index, word in enumerate(self.word_corpus):
            tf_idf = self.word_idf[index] if item.count(word) > 0 else 0
            tf_idf_item.append(tf_idf)
        return tf_idf_item


class BM11Model(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame, k1 = 1.0):
        super().__init__(word_corpus)
        self.metric = "cosine"
        self.k1 = k1
    
    def compute_idf(self, dataset:list):
        word_idf = []
        dataset_size = len(dataset)
        for index, word in enumerate(self.word_corpus):
            word_idf.append(0)
            for item in dataset:
                if word in item:
                    word_idf[index] += 1
            word_idf[index] = math.log((dataset_size - word_idf[index] + 0.5) / (word_idf[index] + 0.5), 2)
        self.word_idf = np.asarray(word_idf)
        
    def compute_average_size(self, dataset:list):
        total_size = 0
        for item in dataset:
            total_size += len(item)
        self.average_size = total_size / len(dataset)
    
    def convert_dataset(self, dataset:list):
        # Calcular o idf para cada palavra.
        self.compute_idf(dataset)
        self.compute_average_size(dataset)
        
        tf_idf_dataset = []
        for item in dataset:
            tf_idf_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(tf_idf_dataset)
    
    def compute_tf(self, word:str, item:list, item_size:int):
        word_count = item.count(word)
        if word_count == 0:
            return 0
        else:
            return ((self.k1 + 1) * word_count) / ((self.k1 * item_size / self.average_size) + word_count) 
    
    def convert_item(self, item:list)-> list:
        tf_idf_item = []
        item_size = len(item)
        for index, word in enumerate(self.word_corpus):
            tf = self.compute_tf(word, item, item_size)
            tf_idf = tf * self.word_idf[index] if item.count(word) > 0 else 0
            tf_idf_item.append(tf_idf)
        return tf_idf_item

class BM15Model(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame, k1 = 1.0):
        super().__init__(word_corpus)
        self.metric = "cosine"
        self.k1 = k1
    
    def compute_idf(self, dataset:list):
        word_idf = []
        dataset_size = len(dataset)
        for index, word in enumerate(self.word_corpus):
            word_idf.append(0)
            for item in dataset:
                if word in item:
                    word_idf[index] += 1
            word_idf[index] = math.log((dataset_size - word_idf[index] + 0.5) / (word_idf[index] + 0.5), 2)
        self.word_idf = np.asarray(word_idf)
        
    def compute_average_size(self, dataset:list):
        total_size = 0
        for item in dataset:
            total_size += len(item)
        self.average_size = total_size / len(dataset)
    
    def convert_dataset(self, dataset:list):
        # Calcular o idf para cada palavra.
        self.compute_idf(dataset)
        self.compute_average_size(dataset)
        
        tf_idf_dataset = []
        for item in dataset:
            tf_idf_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(tf_idf_dataset)
    
    def compute_tf(self, word:str, item:list):
        word_count = item.count(word)
        if word_count == 0:
            return 0
        else:
            return ((self.k1 + 1) * word_count) / (self.k1 + word_count) 
    
    def convert_item(self, item:list)-> list:
        tf_idf_item = []
        for index, word in enumerate(self.word_corpus):
            tf = self.compute_tf(word, item)
            tf_idf = tf * self.word_idf[index] if item.count(word) > 0 else 0
            tf_idf_item.append(tf_idf)
        return tf_idf_item

class BM25Model(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame, k = 1.2, b = 0.60):
        super().__init__(word_corpus)
        self.metric = "cosine"
        self.k = k
        self.b = b
    
    def compute_idf(self, dataset:list):
        word_idf = []
        dataset_size = len(dataset)
        for index, word in enumerate(self.word_corpus):
            word_idf.append(0)
            for item in dataset:
                if word in item:
                    word_idf[index] += 1
            word_idf[index] = math.log((dataset_size - word_idf[index] + 0.5) / (word_idf[index] + 0.5), 2)
        self.word_idf = np.asarray(word_idf)
        
    def compute_average_size(self, dataset:list):
        total_size = 0
        for item in dataset:
            total_size += len(item)
        self.average_size = total_size / len(dataset)
    
    def convert_dataset(self, dataset:list):
        # Calcular o idf para cada palavra.
        self.compute_idf(dataset)
        self.compute_average_size(dataset)
        
        tf_idf_dataset = []
        for item in dataset:
            tf_idf_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(tf_idf_dataset)
    
    def compute_tf(self, word:str, item:list, item_size:int):
        word_count = item.count(word)
        if word_count == 0:
            return 0
        else:
            upper = (self.k + 1) * word_count
            bottom = (self.k * (1 - self.b)) + (self.k * self.b * self.average_size / item_size) + word_count
            return upper / bottom 
    
    def convert_item(self, item:list)-> list:
        tf_idf_item = []
        item_size = len(item)
        for index, word in enumerate(self.word_corpus):
            tf = self.compute_tf(word, item, item_size)
            tf_idf = tf * self.word_idf[index] if item.count(word) > 0 else 0
            tf_idf_item.append(tf_idf)
        return tf_idf_item

class SBertNaiveModel(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = "cosine"
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def convert_dataset(self, dataset:list):
        sentences = [" ".join(item) for item in dataset]
        bert_dataset = self.model.encode(sentences)
        self.dataset = np.asarray(bert_dataset)
    
    def convert_item(self, item:list)-> list:
        return []

class SBertModel(VectorRetrievalModel):
    
    def __init__(self, word_corpus:pd.DataFrame):
        super().__init__(word_corpus)
        self.metric = "cosine"
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.sentence_lenght = 384
    
    def convert_dataset(self, dataset:list):
        bert_dataset = []
        for item in dataset:
            bert_dataset.append(self.convert_item(item))
        self.dataset = np.asarray(bert_dataset)
        np.save("bert_dataset", self.dataset)
    
    def convert_item(self, item:list)-> list:
        if len(item) > self.sentence_lenght:
            sentences = []
            splits = int(math.ceil(len(item)/self.sentence_lenght))
            for i in range(splits):
                start = i*self.sentence_lenght
                end = (i+1)*self.sentence_lenght
                sentences.append(" ".join(item[start:end]))           
            embedding = self.model.encode(sentences)
            embedding = np.sum(embedding, axis=0) / len(sentences)
        else:
            embedding = self.model.encode([" ".join(item)])[0]
        
        return list(embedding)