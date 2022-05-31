import os
import numpy as np
import pandas as pd

# Basic path info
SOURCE_FOLDER = "/home/leolope/Projetos/information-retrieval/datasets"
DATASETS = {
    "train": "mini_newsgroups",
}

if __name__ == "__main__":
    train_df = pd.read_csv("datasets/20news-train-filtered.csv")
    category_dataset = pd.read_csv("datasets/20news-categories.csv")
    for group, label in DATASETS.items():
        categories = os.listdir(os.path.join(SOURCE_FOLDER, label))
        categories.sort()
        data = []
        for category in categories:
            files = os.listdir(os.path.join(
                                SOURCE_FOLDER,
                                label,
                                category))
            files.sort()

            category_id = category_dataset[category_dataset["name"] == category].iloc[0]["id"]

            for filename in files:
                try:
                    doc_id = train_df[(train_df["category_id"] == category_id) & (train_df["document_id"] == int(filename))].iloc[0]["id"]
                except Exception as e:
                    print(category_id, filename)
                    raise e
                data.append([doc_id, filename, category_id])
        df = pd.DataFrame(data, columns=["id", "document_id", "category_id"])
        df.to_csv(os.path.join(SOURCE_FOLDER,"20news-test.csv"),
                index=False)
