import pandas as pd
import pickle
import numpy as np

from utils.evaluation import RetrievalEvaluation
from utils.retrieval import (
    BinaryModel,
    BagOfWordsModel,
    TfIdfModel,
    BM1Model,
    BM11Model,
    BM15Model,
    BM25Model
)

MODELS = {
    "binary": BinaryModel,
    "BoW": BagOfWordsModel,
    "TFIDF": TfIdfModel,
    "BM1": BM1Model,
    "BM11": BM11Model,
    "BM15": BM15Model,
    "BM25": BM25Model,
}

if __name__ == "__main__":
    print("==> Evaluating 20News dataset!\n")
    
    word_corpus = pd.read_csv("datasets/20news-word-corpus-2k.csv")
    test_filter = np.load("datasets/test_filter.npy")
    classes = np.load("datasets/20-news-classes.npy")

    print("\n======================================")
    print(f"Starting evaluation sequence for word corpus of size {len(word_corpus)}!")
    print("======================================\n")
    with open("datasets/20-news-processed-no-singles.pickle", "rb") as f:
        dataset = pickle.load(f)

    for model_name, model_class in MODELS.items():
        print(f"=========== Running {model_name} ===========")
        model = model_class(word_corpus)
        
        print("-> Converting dataset...", end=" ")
        model.convert_dataset(dataset)
        print("Done.")

        print("-> Computing Ranked Lists...", end=" ")
        model.compute_ranked_lists(test_filter=test_filter)
        ranked_lists = model.get_ranked_lists()
        print("Done.")

        print("-> Starting evaluation...")
        evaluation = RetrievalEvaluation(ranked_lists, classes)
        evaluation.evaluate_all()

        print("\n============================================")
    
