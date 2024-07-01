# Check correctness of transformer implementation
# This example code is based on PyTorch's example
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import copy
import os
import time

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from distmljs import tensor_serializer


class TransformerModel(nn.Module):

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


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


def get_batch(source: Tensor, i: int, bptt) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def train_eval(model, bptt, ntokens, device, train_data, val_data):
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 1  # for fast run
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(
            model,
            train_data,
            optimizer,
            scheduler,
            epoch,
            bptt,
            ntokens,
            device,
            criterion,
        )
        val_loss = evaluate(model, val_data, bptt, ntokens, device, criterion)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}"
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()
    return best_model


def train(
    model: nn.Module,
    train_data,
    optimizer,
    scheduler,
    epoch,
    bptt,
    ntokens,
    device,
    criterion,
) -> None:
    model.train()  # turn on train mode
    total_loss = 0.0
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
            )
            total_loss = 0
            start_time = time.time()


def evaluate(
    model: nn.Module, eval_data: Tensor, bptt, ntokens, device, criterion
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def make_sample_grad(model, eval_data, bptt, ntokens, device, criterion):
    model.eval()  # disable dropout
    # remove previous grad
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    data, targets = get_batch(eval_data, 0, bptt)
    output = model(data, src_mask)
    output_flat = output.view(-1, ntokens)
    loss = criterion(output_flat, targets)
    loss.backward()
    grad_dict = {}
    for name, par in model.named_parameters():
        grad_dict[name] = par.grad
    batch_data = {
        "data": data.detach().cpu().to(torch.int32).numpy(),
        "targets": targets.detach().cpu().to(torch.int32).numpy(),
        "src_mask": src_mask.detach().cpu().numpy(),
    }
    return grad_dict, batch_data


def snake2camel(name):
    upper = False
    cs = []
    for c in name:
        if c == "_":
            upper = True
            continue
        if upper:
            c = c.upper()
        cs.append(c)
        upper = False
    return "".join(cs)


def map_weight_to_distmljs(torch_weights):
    new_dict = {}
    for k, v in torch_weights.items():
        if k == "pos_encoder.pe":
            continue
        if "in_proj" in k:
            qkv = v.chunk(3)
            # in_proj_weight => in_proj_k.weight
            new_dict[k.replace("in_proj_", "in_proj_q.")] = qkv[0]
            new_dict[k.replace("in_proj_", "in_proj_k.")] = qkv[1]
            new_dict[k.replace("in_proj_", "in_proj_v.")] = qkv[2]
        else:
            new_dict[k] = v
    return {snake2camel(k): v.detach().cpu().numpy() for k, v in new_dict.items()}


def main():
    print(
        "This sample trains transformer model with PyTorch and saves data to check correctness of gradient computation in distmljs."
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 20
    eval_batch_size = 10
    # shape [seq_len, batch_size]
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)
    bptt = 10

    ntokens = len(vocab)  # size of vocabulary
    emsize = 64  # embedding dimension
    d_hid = 64  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    best_model = train_eval(model, bptt, ntokens, device, train_data, val_data)

    criterion = nn.CrossEntropyLoss()

    best_model_weights = best_model.state_dict()
    grads, sample_batch = make_sample_grad(
        model, test_data, bptt, ntokens, device, criterion
    )
    print("saving trained model to pytorch_trained_model.pt")
    torch.save(best_model_weights, os.path.join(out_dir, "pytorch_trained_model.pt"))
    print("converting weight for distmljs to pytorch_trained_weight.bin")
    with open(os.path.join(out_dir, "pytorch_trained_weight.bin"), "wb") as f:
        f.write(
            tensor_serializer.serialize_tensors_to_bytes(
                map_weight_to_distmljs(best_model_weights)
            )
        )
    print("saving a batch to pytorch_sample_batch.bin")
    with open(os.path.join(out_dir, "pytorch_sample_batch.bin"), "wb") as f:
        f.write(tensor_serializer.serialize_tensors_to_bytes(sample_batch))
    print("saving gradient of a batch to pytorch_trained_grad.bin")
    with open(os.path.join(out_dir, "pytorch_trained_grad.bin"), "wb") as f:
        f.write(
            tensor_serializer.serialize_tensors_to_bytes(map_weight_to_distmljs(grads))
        )
    print(
        "open http://localhost:8080/sample/transformer/output/gradient_check.html to run gradient check."
    )


if __name__ == "__main__":
    main()
