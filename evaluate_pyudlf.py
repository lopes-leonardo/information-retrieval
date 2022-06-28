import pandas as pd
import pickle
import numpy as np

from sklearn import metrics
from utils.evaluation import RetrievalEvaluation
from utils.retrieval import SBertModel

embeddings = np.load("transencoder_dataset.npy", allow_pickle=True)
print(embeddings.shape)
distances = metrics.pairwise.cosine_distances(embeddings, embeddings)
ranked_lists = []
for item in distances:
    rank_map = np.argsort(item)
    ranked_lists.append(rank_map)
ranked_lists = np.asarray(ranked_lists)
# output_ranked_lists = ranked_lists[:,:1200]
# print(output_ranked_lists.shape)

# with open("datasets/20-news-transencoder-ranked-lists.txt", "w") as f:
#     for ranked_list in output_ranked_lists:
#         f.write(" ".join(ranked_list.astype(str)))
#         f.write("\n")
#     f.close()
# print(ranked_lists[:,:1200].shape)
# exit()

classes = np.load("datasets/20-news-classes.npy")
# ranked_lists = np.load("output/20-news-sbert-lhrr-200.npy", allow_pickle=True)
# test_filter = np.load("datasets/test_filter.npy")

evaluation = RetrievalEvaluation(ranked_lists, classes)
evaluation.evaluate_all()


