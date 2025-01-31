import spacy
import torchtext.vocab
import datasets

from src.constants import SENTENCE, SENTENCE_IDS, SENTENCE_TOKENS


# Tokenization function
def tokenize_example(
        example, 
        language: spacy.Language, 
        max_length: int, 
        lower: bool, 
        sos_token: str, 
        eos_token: str
) -> dict[str, list[str]]:
    tokens = [token.text for token in language.tokenizer(example[SENTENCE])][:max_length]
    if lower:
        tokens = [token.lower() for token in tokens]
    tokens = [sos_token] + tokens + [eos_token]
    return {SENTENCE_TOKENS: tokens}


# Apply tokenization to datasets
def tokenize_data(
    data: datasets.Dataset, 
    language: spacy.Language, 
    max_length: int, 
    lower: bool, 
    sos_token: str, 
    eos_token: str
) -> datasets.Dataset:
    fn_kwargs = {"language": language, "max_length": max_length, "lower": lower, "sos_token": sos_token, "eos_token": eos_token}
    return data.map(tokenize_example, fn_kwargs=fn_kwargs)


# Vocabulary creation function
def build_vocab(dataset: datasets.Dataset, special_tokens: list[str], min_freq: int):
    return torchtext.vocab.build_vocab_from_iterator(
        dataset[SENTENCE_TOKENS], min_freq=min_freq, specials=special_tokens
    )


# Numericalization function
def numericalize_example(example, vocab: torchtext.vocab.Vocab) -> dict[str, list[int]]:
    ids = vocab.lookup_indices(example[SENTENCE_TOKENS])
    return {SENTENCE_IDS: ids}


# Apply numericalization to datasets
def numericalize_data(dataset: datasets.Dataset, vocab: torchtext.vocab.vocab):
    fn_kwargs = {"vocab": vocab}
    return dataset.map(numericalize_example, fn_kwargs=fn_kwargs)


# Set the data format for PyTorch
def set_data_format(dataset: datasets.Dataset, format_type="torch", format_columns=[SENTENCE_IDS]):
    return dataset.with_format(
        type=format_type,
        columns=format_columns,
        output_all_columns=True
    )
