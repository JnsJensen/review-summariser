import re
from enum import Enum
import contractions as ct
import pandas as pd
import polars as pl
from transformers import GPT2Tokenizer
import math
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import numpy as np

class Paths:
    data_dir = "datasets/"

    # Dataset paths
    data_original = data_dir + "full/" # Use this for full dataset that contains all review matadata
    data_pruned = data_dir + "pruned/" # Only datasets with one number to the right side of it have a pruned version
    data_tokenized = data_dir + "tokenized/" # Only datasets with two numbers to the right side of it have a tokenized version

    full = "All_Amazon_Review_5" # 80 GB
    arts = "Arts_Crafts_and_Sewing" # 518 MB / 629 MB / 1.18 GB
    video = "Amazon_Instant_Video_5" # 28 MB
    gift = "Gift_Cards_5" # 0.88 MB

PAD_token = "<|padding|>"
BOS_token = "<|startoftext|>"
EOS_token = "<|endoftext|>"
UNK_token = "<|unknown|>"

"""
DatasetType enum
- ORIGINAL: The original json dataset
- PRUNED: The pruned csv dataset
- TOKENIZED: The tokenized csv dataset
"""
class DatasetType(Enum):
    ORIGINAL = 0
    PRUNED = 1
    TOKENIZED = 2

class MaxTokenLength():
    ARTS_REVIEW = 206
    ARTS_SUMMARY = 37
    VIDEO_REVIEW = 4064
    VIDEO_SUMMARY = 61
    GIFT_REVIEW = 553
    GIFT_SUMMARY = 27

listify = lambda x: x.split("|")
stringify = lambda x: "|".join(list(x.cast(pl.Utf8)))

"""
Normalizes strings by:
- Removing double spaces
- Converting to lower case
- Expanding contractions
"""
def normalize_text(text: str) -> str:
    # Lowercase text
    text = text.lower()
    # Expand contractions
    text = ct.fix(text)
    # Remove non-ascii characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove all symbols
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading and trailing spaces
    text = text.strip()
    return text

"""
Normalizes the reviewText and summary columns of a dataframe by:
- Using normalize_text on the 'reviewText' and 'summary' columns
"""
def normalize_text_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.lazy().select([
        pl.col("reviewText").apply(normalize_text),
        pl.col("summary").apply(normalize_text),
        pl.exclude(["reviewText", "summary"])
    ]).collect()
    return df

"""
Prunes the dataset by:
- Removing reviews with a summary of "five stars", "four stars", "three stars", "two stars", or "one star
- Normalizing the reviewText and summary columns by the use of normalize_text_df
- Converting the overall column to a float between 0 and 1
"""
def prune(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    # list of unwanted summaries in lower case
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df.dropna())
    assert(isinstance(df, pl.DataFrame))

    df = normalize_text_df(df)

    df = df.filter(pl.col("reviewText").str.split(" ").apply(len) > 15) \
           .filter(pl.col("reviewText").str.split(" ").apply(len) < 100) \
           .filter(pl.col("summary").str.split(" ").apply(len) > 5) \
           .filter(pl.col("summary").str.split(" ").apply(len) < 15)

    df = df.lazy().select([
        pl.col("overall").apply(lambda x: int(x)/5),
        pl.exclude(["overall"])
    ]).collect()
    return df.select([pl.col("reviewText"), pl.col("summary"), pl.col("overall")])

"""
Writes a dataframe to a csv file
- Distinguishes between polars and pandas dataframes
"""
def write_to_csv(df: pd.DataFrame | pl.DataFrame, path: str) -> None:
    if isinstance(df, pd.DataFrame):
        df.to_csv(path + ".csv", index=False)
    elif isinstance(df, pl.DataFrame):
        df.write_csv(path + ".csv")

"""
Tokenizes the reviewText and summary columns of a dataframe by:
- Using the tokenizer to encode the 'reviewText' and 'summary' columns
"""
def tokenize(df: pl.DataFrame, tokenizer: GPT2Tokenizer) -> pl.DataFrame:
    t = lambda x: tokenizer.encode(x, add_special_tokens=True)
    df = df.lazy().select([
        pl.col("reviewText").apply(t),
        pl.col("summary").apply(t),
        pl.exclude(["reviewText", "summary"])
    ]).collect()
    return df

"""
Loads a dataset from a csv file
- Distinguishes between polars and pandas dataframes
- Distinguishes between the original, pruned, and tokenized datasets
"""
def load_dataset(dataset: str, dataset_type: DatasetType, keep_cols = ["reviewText", "summary", "overall"]) -> pd.DataFrame | pl.DataFrame:
    if dataset_type == DatasetType.ORIGINAL:
        # return pl.read_json(data_original + dataset + ".json", json_lines=True).select([keep_cols])
        return pd.read_json(Paths.data_original + dataset + ".json", lines=True)[keep_cols]
    elif dataset_type == DatasetType.PRUNED:
        return pl.read_csv(Paths.data_pruned + dataset + ".csv", dtypes={"reviewtext": pl.Utf8, "summary": pl.Utf8, "overall": pl.Float32})
    elif dataset_type == DatasetType.TOKENIZED:
        df = pl.read_csv(Paths.data_tokenized + dataset + ".csv", dtypes={"reviewtext": pl.Utf8, "summary": pl.Utf8, "overall": pl.Float32})
        
        df = df.lazy().select([
            pl.col("reviewText").apply(listify).cast(pl.List(pl.Int64)),
            pl.col("summary").apply(listify).cast(pl.List(pl.Int64)),
            pl.exclude(["reviewText", "summary"])
        ]).collect()
        return df

"""
Saves a dataset to a csv file
- Distinguishes between the pruned and tokenized datasets
"""
def save_dataset(df: pl.DataFrame, dataset: str, dataset_type: DatasetType) -> None:
    if dataset_type == DatasetType.PRUNED:
        write_to_csv(df, Paths.data_pruned + dataset)
    elif dataset_type == DatasetType.TOKENIZED:
        
        df = df.lazy().select([
            pl.col("reviewText").apply(stringify),
            pl.col("summary").apply(stringify),
            pl.exclude(["reviewText", "summary"])
        ]).collect()
        write_to_csv(df, Paths.data_tokenized + dataset)

"""
Preprocesses a dataset by:
- Loading the dataset
- Pruning the dataset
- Tokenizing the dataset
"""
def preprocess(dataset: str, dataset_type: DatasetType, tokenizer, keep_cols = ["reviewText", "summary", "overall"], save_steps=True) -> pd.DataFrame | pl.DataFrame:
    if dataset_type == DatasetType.ORIGINAL:
        df = load_dataset(dataset, dataset_type, keep_cols)
        df = prune(df)
        if save_steps:
            save_dataset(df, dataset, DatasetType.PRUNED)
        df = tokenize(df, tokenizer)
        if save_steps:
            save_dataset(df, dataset, DatasetType.TOKENIZED)
        return df
    elif dataset_type == DatasetType.PRUNED:
        df = load_dataset(dataset, dataset_type)
        df = tokenize(df, tokenizer)
        if save_steps:
            save_dataset(df, dataset, DatasetType.TOKENIZED)
        return df
    elif dataset_type == DatasetType.TOKENIZED:
        df = load_dataset(dataset, dataset_type)
        return df

class Modifiers:
    """
    Class of escape sequences for color printing
    """
    class Colors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BLACK = '\033[30m'
        MAGENTA = '\033[35m'
    
    class Styles:
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        ITALIC = '\033[3m'

    ENDC = '\033[0m'


def get_run_dir() -> str:
    # keep track of the count
    count = 0
    try:
        # read previous count from file
        with open("count.txt", "r") as f:
            count = int(f.read())
    except:
        print("No count file found. Reading from runs/ folder...")
        # check the first three character of each file in runs/ folder, and initialise count to the highest number found
        found_file = False
        for file in os.listdir("runs/"):
            if file[:3].isdigit():
                count = max(count, int(file[:3]))
                found_file = True
        
        if not found_file:
            print("No file found starting with digits in runs/ folder. Starting from 0...")

    count += 1

    # write count to file
    with open("count.txt", "w") as f:
        f.write(str(count))

    return f"runs/{count:03d}"

def get_data_loaders(dataset, batch_size, split_ratios):
    """
    Returns the data loaders for the train, validate and test datasets
    """
    num_reviwes = len(dataset)
    n_train_dataset = math.floor(num_reviwes * split_ratios[0])
    n_validate_dataset = math.floor(num_reviwes * split_ratios[1])
    n_test_dataset = num_reviwes - n_train_dataset - n_validate_dataset

    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train_dataset, n_validate_dataset, n_test_dataset])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

"""
Modified printing for the terminal
"""
def print_mod(text: str, modifiers: list) -> None:
    print("".join(modifiers) + text + Modifiers.ENDC)

"""
Get a bounded list of colors from a colormap
"""
def get_colors_from_cmap(num_colors: int, cmap, color_lims=[]):
    # if color limits are empty, then use a range proportional to the number of colors
    # the higher the number of colors, the larger the range, approaching the full range of the colormap
    if len(color_lims) == 0:
        percentage_range = (1 - (1 / num_colors)) * 0.75
        color_lims = [0.5 * (1 - percentage_range), 0.5 * (1 + percentage_range)]
    
    return [cmap(i) for i in np.linspace(color_lims[0], color_lims[1], num_colors)]