"""
Webブラウザを接続した分散計算を行う。
環境変数で設定を行う。
MODEL: mlp, convのいずれか。モデルの種類を指定する。
N_CLIENTS: 分散計算に参加するクライアント数。1以上の整数を指定する。指定しない場合は1が指定されたとみなす。

実行はuvicorn経由で行う。コマンド例(Mac/Linuxの場合):
MODEL=conv N_CLIENTS=2 npm run train

Windowsの場合はsetコマンドを使用して以下のようになる:
set MODEL=conv
set N_CLIENTS=2
npm run train
"""

import asyncio
import math
from uuid import uuid4
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from kakiage.server import KakiageServerWSConnectEvent, KakiageServerWSReceiveEvent, setup_server
from kakiage.tensor_serializer import serialize_tensors_to_bytes, deserialize_tensor_from_bytes
from sample_net import make_net

# スクリプトの配布
kakiage_server = setup_server()
app = kakiage_server.app

# PyTorchを用いた初期モデルの作成、学習したモデルのサーバサイドでの評価


def test(model, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for image, label in loader:
            logit = model(image)
            loss = F.cross_entropy(logit, label, reduction="sum")
            loss_sum += loss.item()
            pred = logit.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            count += len(image)

    return {"loss": loss_sum / count, "accuracy": correct / count}


def get_dataset_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../pytorch_data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../pytorch_data', train=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader


async def main():
    print("MNIST data parallel training sample")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(0)
    lr = 0.01
    n_client_wait = int(os.environ.get("N_CLIENTS", "1"))
    model_name = os.environ.get("MODEL", "mlp")
    print(f"Model: {model_name}")
    model = make_net(model_name)
    train_loader, test_loader = get_dataset_loader()
    client_ids = []
    print(f"Waiting {n_client_wait} clients to connect")

    async def get_event():
        while True:
            event = await kakiage_server.event_queue.get()
            if isinstance(event, KakiageServerWSConnectEvent):
                client_ids.append(event.client_id)
                return None
            else:
                return event
    while len(client_ids) < n_client_wait:
        await get_event()

    test_results = []
    for epoch in range(2):
        print(f"epoch {epoch}")
        with torch.no_grad():
            weights = {}
            for k, v in model.state_dict().items():
                # fc1.weight, fc1.bias, ...
                weights[k] = v.detach().numpy()
            for image, label in train_loader:
                item_ids_to_delete = []
                weight_item_id = uuid4().hex
                kakiage_server.blobs[weight_item_id] = serialize_tensors_to_bytes(
                    weights)
                item_ids_to_delete.append(weight_item_id)
                # TODO: 切断対応(今は切断されると待ち続ける)
                batch_size = len(image)
                n_clients = len(client_ids)
                chunk_size = math.ceil(batch_size / n_clients)
                chunk_sizes = []
                grad_item_ids = []
                for c, client_id in enumerate(client_ids):
                    image_chunk = image[c*chunk_size:(c+1)*chunk_size]
                    label_chunk = label[c*chunk_size:(c+1)*chunk_size]
                    chunk_sizes.append(len(image_chunk))
                    dataset_item_id = uuid4().hex
                    kakiage_server.blobs[dataset_item_id] = serialize_tensors_to_bytes(
                        {
                            "image": image_chunk.detach().numpy().astype(np.float32),
                            "label": label_chunk.detach().numpy().astype(np.int32),
                        }
                    )
                    item_ids_to_delete.append(dataset_item_id)
                    grad_item_id = uuid4().hex
                    await kakiage_server.send_message(client_id, {
                        "model": model_name,
                        "weight": weight_item_id,
                        "dataset": dataset_item_id,
                        "grad": grad_item_id
                    })
                    grad_item_ids.append(grad_item_id)
                    item_ids_to_delete.append(grad_item_id)
                complete_count = 0
                while True:
                    event = await get_event()
                    if isinstance(event, KakiageServerWSReceiveEvent):
                        complete_count += 1
                        if complete_count >= n_clients:
                            break
                    else:
                        print("unexpected event")

                # calculate weighted average of gradients
                grad_arrays = {}
                for chunk_size, grad_item_id in zip(chunk_sizes, grad_item_ids):
                    chunk_weight = chunk_size / batch_size
                    chunk_grad_arrays = deserialize_tensor_from_bytes(
                        kakiage_server.blobs[grad_item_id])
                    for k, v in chunk_grad_arrays.items():
                        if k in grad_arrays:
                            grad_arrays[k] += v * chunk_weight
                        else:
                            grad_arrays[k] = v * chunk_weight
                for k, v in weights.items():
                    grad = grad_arrays[k]
                    v -= lr * grad
                for item_id in item_ids_to_delete:
                    del kakiage_server.blobs[item_id]
        for k, v in model.state_dict().items():
            v[:] = torch.from_numpy(weights[k])
        test_result = test(model, test_loader)
        print(test_result)
        test_results.append(test_result)
    torch.save(model.state_dict(), os.path.join(
        output_dir, "kakiage_trained_model.pt"))
    with open(os.path.join(output_dir, "kakiage_training.pkl"), "wb") as f:
        pickle.dump({"test_results": test_results}, f)
    print("training ended")


asyncio.get_running_loop().create_task(main())
