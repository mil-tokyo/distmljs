# Check correctness of transformer implementation
# This example code is based on PyTorch's example
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import os

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

import torch
from torch import Tensor
from torch.utils.data import dataset

from distmljs import tensor_serializer


def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int, device) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def main():
    print("This sample downloads and convert WikiText2 dataset for distmljs.")
    out_dir = os.path.join("output", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    device = "cpu"

    batch_size = 20
    eval_batch_size = 10
    # shape [seq_len, batch_size]
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"saving {name} data for distmljs to wikitext2_{name}.bin")
        with open(os.path.join(out_dir, f"wikitext2_{name}.bin"), "wb") as f:
            f.write(
                tensor_serializer.serialize_tensors_to_bytes(
                    {"data": data.to(torch.int32).numpy()}
                )
            )
    print("open http://localhost:8080/sample/transformer/output/index.html to train.")


if __name__ == "__main__":
    main()
