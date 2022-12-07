import re
from enum import Enum
import contractions as ct
import pandas as pd
import polars as pl
from transformers import GPT2Tokenizer

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

    PAD_token = "<|pad|>"
    BOS_token = "<|bos|>"
    EOS_token = "<|eos|>"
    UNK_token = "<|unk|>"

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

listify = lambda x: x.split("|")
stringify = lambda x: "|".join(list(x.cast(pl.Utf8)))

"""
Normalizes strings by:
- Removing double spaces
- Converting to lower case
- Expanding contractions
"""
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", ct.fix(text.lower()))

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

    df = df.filter(pl.col("summary") != "five stars") \
           .filter(pl.col("summary") != "four stars") \
           .filter(pl.col("summary") != "three stars") \
           .filter(pl.col("summary") != "two stars") \
           .filter(pl.col("summary") != "one star")
    
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