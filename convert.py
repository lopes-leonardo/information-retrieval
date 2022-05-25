import os
import numpy as np
import pandas as pd

# Basic path info
SOURCE_FOLDER = "/home/leolope/Projetos/information-retrieval/datasets"
DATASETS = {
    "train": "20_newsgroups",
}

if __name__ == "__main__":
    for group, label in DATASETS.items():
        categories = os.listdir(os.path.join(SOURCE_FOLDER, label))
        categories.sort()
        data = []
        for i, category in enumerate(categories):
            files = os.listdir(os.path.join(
                                SOURCE_FOLDER,
                                label,
                                category))
            files.sort()
            for filename in files:
                try:
                    f = open(os.path.join(
                                SOURCE_FOLDER,
                                label,
                                category,
                                filename),
                                "r",
                                encoding="ISO-8859-1")                      
                    text = ""
                    is_header = True
                    seen_lines = False
                    for line in f:
                        if line.strip() == "":
                            continue
                        
                        if is_header and ":" in line:
                            header = line.split(":")
                            if header[0].lower() == "lines":
                                seen_lines = True
                            
                            if header[0].lower() == "subject":
                                text += " " + header[1].strip()
                            continue

                        if seen_lines:
                            is_header = False
                        
                        text += " " + line.strip()
                except Exception as e:
                    print(filename, category)
                    raise(e)
                data.append([filename, text.strip(), i])
        df = pd.DataFrame(data, columns=["id", "text", "category_id"])
        df.to_csv(os.path.join(SOURCE_FOLDER,f"20news-{group}-filtered.csv"),
                index=False)
