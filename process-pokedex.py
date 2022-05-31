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


class PokedexEvaluation(RetrievalEvaluation):
    
    def __init__(self, ranked_lists:np.ndarray, classes:np.ndarray, use_both_types:bool = False):
        super().__init__(ranked_lists, classes)
        self.use_both_types = use_both_types
        
    def p_at_n(self, n:int)->float:
        p_total = 0
        for rank in self.ranked_lists:
            p = 0
            target_class = self.classes[0][rank[0]]
            secundary_class = self.classes[1][rank[0]]
            for i in range(n):
                if not self.use_both_types:
                    if self.classes[0][rank[i]] == target_class:
                        p += 1
                else:
                    if self.classes[0][rank[i]] == target_class:
                        p += 1
                    elif self.classes[0][rank[i]] == secundary_class and secundary_class is not None:
                        p += 1
                    elif self.classes[1][rank[i]] == target_class:
                        p += 1
                    elif self.classes[1][rank[i]] == secundary_class and secundary_class is not None:
                        p += 1
            p = p / n
            p_total += p
        return p_total / self.n
    
    def evaluate_all(self) -> None:
        print("=========== Evaluation Procedure ===========")
        print("Evaluation dataset size:", self.n)
        print("Precision at 10:", self.p_at_10())
        print("Precision at 20:", self.p_at_20())
        print("Precision at 50:", self.p_at_50())
        print("Precision at 100:", self.p_at_100())

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
    print("==> Evaluating Pokédex dataset!\n")
    word_corpus = pd.read_csv("datasets/pokedex-word-corpus-1903.csv")
    classes = np.load("datasets/pokedex-classes.npy", allow_pickle=True)

    print("\n======================================")
    print(f"Starting evaluation sequence for word corpus of size {len(word_corpus)}!")
    print("======================================\n")
    with open("datasets/pokedex-processed.pickle", "rb") as f:
        dataset = pickle.load(f)

    for model_name, model_class in MODELS.items():
        print(f"=========== Running {model_name} ===========")
        model = model_class(word_corpus)
        
        print("-> Converting dataset...", end=" ")
        model.convert_dataset(dataset)
        print("Done.")

        print("-> Computing Ranked Lists...", end=" ")
        model.compute_ranked_lists()
        ranked_lists = model.get_ranked_lists()
        print("Done.")

        print("-> Starting evaluation...")
        evaluation = PokedexEvaluation(ranked_lists, classes, use_both_types=True)
        evaluation.evaluate_all()

        print("\n============================================")
    
