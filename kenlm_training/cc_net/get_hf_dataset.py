import cld3
import random
import nltk
from typing import Optional

import func_argparse
from datasets import load_dataset
from tqdm import tqdm

from cc_net import text_normalizer

sent_tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")

random.seed(1234)

def text_normalizer_list(text, accent, case, numbers, punct):
    text_list = sent_tokenizer.tokenize(text)
    norm_text_list = []
    for text in text_list:
        norm_text_list.append(text_normalizer.normalize(text, accent=accent, case=case, numbers=numbers, punct=punct))
    return norm_text_list

def dl(
    dataset: str,
    output_file: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = True,
    accent: bool = False,
    case: bool = False,
    numbers: bool = True,
    punct: int = 1,
    max_docs: Optional[int] = None,
    seed: int = 0,
    buffer_size: int = 10000,
    text_key: str = "TEXT",
    buffer_shuffle: bool = False
):
    """Download dataset from the Hugging Face hub."""
    dataset = load_dataset(
       dataset,
       name=name,
       data_dir=data_dir,
       data_files=data_files,
       split="train",
       streaming=streaming,
    )
    norm_text_key = text_key + "_norm"
    dataset_norm = dataset.map(
        lambda x: {norm_text_key : text_normalizer_list(
            x[text_key], accent=accent, case=case, numbers=numbers, punct=punct
        )}
    )

    if buffer_shuffle:
        dataset_norm = dataset_norm.shuffle(buffer_size=buffer_size, seed=seed)

    count = 0
    with open(output_file, "w") as o:
        with tqdm(total=max_docs) as pbar:
            for doc in dataset_norm:
                doc = doc[norm_text_key]

                pbar.update(1)

                if doc is None:
                    continue

                for sent in doc:
                    sent = sent.rstrip("\n")
                    print(sent, file=o)

                count += 1
                if max_docs and count == max_docs:
                    break

    # shuffle the sentences later
    with open(output_file, "r") as f:
        lines = f.readlines()

    # filter lines
    print ("Filtering lines for english")
    filtered_lines = []
    for line in tqdm(lines):
        lang = cld3.get_language(line).language

        if lang == 'en':
            filtered_lines.append(line)

    # shuffle and write to disk
    print ("Shuffle and write to disk")
    random.shuffle(filtered_lines)

    with open(output_file, "w") as f:
        for line in tqdm(filtered_lines):
            line = line.rstrip("\n")
            print (line, file=f)

if __name__ == "__main__":
    func_argparse.main(dl)
