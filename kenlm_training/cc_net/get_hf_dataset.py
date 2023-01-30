from typing import Optional

import func_argparse
from datasets import load_dataset
from tqdm import tqdm

from cc_net import text_normalizer

def text_normalizer_list(text_list, accent, case, numbers, punct):
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
    streaming: bool = False,
    accent: bool = False,
    case: bool = False,
    numbers: bool = True,
    punct: int = 1,
    max_docs: Optional[int] = None,
    seed: int = 0,
    buffer_size: int = 10000,
    text_key: str = "TEXT"
):
    """Download dataset from the Hugging Face hub."""
    dataset = load_dataset(
        dataset,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        split=split,
        streaming=streaming,
    )

    norm_text_key = text_key + "_norm"
    dataset_norm = dataset.map(
        lambda x: {norm_text_key : text_normalizer_list(
            x[text_key], accent=accent, case=case, numbers=numbers, punct=punct
        )}
    )

    print (next(iter(dataset_norm)).keys())
    dataset_norm = dataset_norm.shuffle(seed=seed)
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


if __name__ == "__main__":
    func_argparse.main(dl)
