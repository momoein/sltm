import os
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets

from src.constants import VIDEO, SENTENCE


def load_and_process_csv(file_path: str, delimiter: str = "\t"):
    """
    Load and process the CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with columns 'VIDEO' and 'SENTENCE'.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    return pd.DataFrame({
        VIDEO: [f"{video}.mp4" for video in df["SENTENCE_NAME"]],
        SENTENCE: df["SENTENCE"],
    })


def filter_existing_videos(df, dir_path):
    """
    Remove entries with missing video files from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with video file paths.
        dir_path (str): Directory containing video files.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    existing_videos = df[VIDEO].apply(lambda video: os.path.exists(os.path.join(dir_path, video)))
    return df[existing_videos]


def split_dataset(df, train_size, seed):
    """
    Split the dataset into training and validation subsets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        train_size (float): Proportion of the data to include in the training split.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Training and validation subsets as DataFrames.
    """
    x, _, y, _ = train_test_split(df[VIDEO], df[SENTENCE], train_size=train_size, random_state=seed)
    return pd.DataFrame({VIDEO: x, SENTENCE: y})


def create_huggingface_datasets(train_df, valid_df, test_df):
    """
    Convert DataFrames to Hugging Face Dataset format.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        valid_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.

    Returns:
        tuple: Hugging Face datasets for training, validation, and testing.
    """
    train_data = datasets.Dataset.from_dict(train_df.to_dict(orient="list"))
    valid_data = datasets.Dataset.from_dict(valid_df.to_dict(orient="list"))
    test_data = datasets.Dataset.from_dict(test_df.to_dict(orient="list"))
    return train_data, valid_data, test_data


def get_datasets(files_paths: dict, video_dirs: dict, train_size: dict, seed) -> dict[str, datasets.Dataset]:
    # Load datasets
    train_df = load_and_process_csv(files_paths["train"])
    valid_df = load_and_process_csv(files_paths["val"])
    test_df = load_and_process_csv(files_paths["test"])

    # Filter datasets
    train_df = filter_existing_videos(train_df, video_dirs["train"])
    valid_df = filter_existing_videos(valid_df, video_dirs["val"])
    test_df = filter_existing_videos(test_df, video_dirs["test"])

    # Subset selection
    train_df = split_dataset(train_df, train_size["train"], seed)
    valid_df = split_dataset(valid_df, train_size["val"], seed)
    test_df = split_dataset(test_df, train_size["test"], seed)

    # Convert to Hugging Face datasets
    train_data, valid_data, test_data = create_huggingface_datasets(train_df, valid_df, test_df)

    return {"train": train_data, "val": valid_data, "test": test_data}

