from typing import Optional

import func_argparse
from datasets import load_dataset
from tqdm import tqdm

from cc_net import text_normalizer


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
        lambda x: {norm_text_key : text_normalizer.normalize(
            x[text_key], accent=accent, case=case, numbers=numbers, punct=punct
        )}
    )

    print (next(iter(dataset_norm)))
    dataset_norm = dataset_norm.shuffle(buffer_size=buffer_size, seed=seed)
    count = 0
    with open(output_file, "w") as o:
        with tqdm(total=max_docs) as pbar:
            for doc in dataset_norm:
                doc = doc[norm_text_key]

                if doc is None:
                    continue

                doc = doc.rstrip("\n")
                count += 1
                print(doc, file=o)
                if max_docs and count == max_docs:
                    break
                pbar.update(1)


if __name__ == "__main__":
    func_argparse.main(dl)
