import pandas as pd
import re

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_text_feature(df):
    df["text"] = df["titre"] + " " + df["recette"]
    df["text"] = df["text"].apply(clean_text)
    return df


def encode_labels(df):
    label_map = {
        "Entrée": 0,
        "Plat principal": 1,
        "Dessert": 2
    }
    df["label"] = df["type"].map(label_map)
    return df



def preprocess(path):
    df = load_data(path)
    df = create_text_feature(df)
    df = encode_labels(df)

    return df[["text", "label"]]

def save_data(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = "train.csv"
    output_path = "processed.csv"

    df = preprocess(input_path)
    save_data(df, output_path)

    print("Preprocessing completed.")
    print(df.head())