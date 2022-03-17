"""
Webブラウザを接続した分散計算を行う。学習したモデルはONNX形式で出力され、推論用フレームワークで使用できる。
環境変数で設定を行う。
MODEL: mlp, conv, resnet18のいずれか。モデルの種類を指定する。
N_CLIENTS: 分散計算に参加するクライアント数。1以上の整数を指定する。指定しない場合は1が指定されたとみなす。
EPOCH: 学習エポック数。デフォルトは2。
BATCH_SIZE: バッチサイズ。全クライアントの合計。デフォルトは32。

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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from kakiage.server import KakiageServerWSConnectEvent, KakiageServerWSReceiveEvent, setup_server
from kakiage.tensor_serializer import serialize_tensors_to_bytes, deserialize_tensor_from_bytes
from sample_net import make_net, get_io_shape, get_dataset_loader

# setup server to distribute javascript and communicate
kakiage_server = setup_server()
app = kakiage_server.app

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


def snake2camel(name):
    """
    running_mean -> runningMean
    PyTorch uses snake_case, kakiage uses camelCase
    """
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


def is_trainable_key(name):
    if "running" in name:
        # runningMean, runningVar
        return False
    if "numBatchesTracked" in name:
        return False
    return True


async def main():
    print("MNIST / CIFAR data parallel training sample")
    n_client_wait = int(os.environ.get("N_CLIENTS", "1"))
    n_epoch = int(os.environ.get("EPOCH", "2"))
    dataset_name = os.environ.get("DATASET", "mnist")
    model_name = os.environ.get("MODEL", "mlp")
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    input_shape, n_classes = get_io_shape(dataset_name)
    output_dir = os.path.join("results", model_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(0)
    lr = 0.01
    print(
        f"Model: {model_name}, dataset: {dataset_name}, batch size: {batch_size}")
    model = make_net(model_name, input_shape, n_classes)
    train_loader, test_loader = get_dataset_loader(dataset_name, batch_size)
    client_ids = []
    print(f"Waiting {n_client_wait} clients to connect")

    # Gets server event
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
        print(f"{len(client_ids)} / {n_client_wait} clients connected")

    test_results = []
    start_time = time.time()
    for epoch in range(n_epoch):
        print(f"epoch {epoch} / {n_epoch}")
        with torch.no_grad():
            weights = {}
            for k, v in model.state_dict().items():
                # fc1.weight, fc1.bias, ...
                vnum = v.detach().numpy()
                if vnum.dtype == np.int64:
                    vnum = vnum.astype(np.int32)
                weights[snake2camel(k)] = vnum
            for i, (image, label) in enumerate(train_loader):
                print(
                    f"\riter {i} / {len(train_loader)} elapsed={int(time.time() - start_time)}s", end="")
                item_ids_to_delete = []
                weight_item_id = uuid4().hex
                kakiage_server.blobs[weight_item_id] = serialize_tensors_to_bytes(
                    weights)
                item_ids_to_delete.append(weight_item_id)
                # 切断・クライアントの動的追加への対応はなされていない(今は切断されると待ち続ける)
                batch_size = len(image)
                n_clients = len(client_ids)
                chunk_size = math.ceil(batch_size / n_clients)
                chunk_sizes = []
                grad_item_ids = []
                # split batch into len(client_ids) chunks
                for c, client_id in enumerate(client_ids):
                    image_chunk = image[c*chunk_size:(c+1)*chunk_size]
                    label_chunk = label[c*chunk_size:(c+1)*chunk_size]
                    chunk_sizes.append(len(image_chunk))
                    dataset_item_id = uuid4().hex
                    # set blob (binary data) in server so that client can download by spceifying id
                    kakiage_server.blobs[dataset_item_id] = serialize_tensors_to_bytes(
                        {
                            "image": image_chunk.detach().numpy().astype(np.float32),
                            "label": label_chunk.detach().numpy().astype(np.int32),
                        }
                    )
                    item_ids_to_delete.append(dataset_item_id)
                    grad_item_id = uuid4().hex
                    # send client to calculate gradient given the weight and batch
                    await kakiage_server.send_message(client_id, {
                        "model": model_name,
                        "inputShape": list(input_shape),
                        "nClasses": n_classes,
                        "weight": weight_item_id,
                        "dataset": dataset_item_id,
                        "grad": grad_item_id
                    })
                    grad_item_ids.append(grad_item_id)
                    item_ids_to_delete.append(grad_item_id)
                complete_count = 0
                # Wait for all clients to complete
                # No support for disconnection and dynamic addition of clients (this implementation waits disconnected client forever)
                # To support, handle event such as KakiageServerWSConnectEvent
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
                    if is_trainable_key(k):
                        # update weight using SGD (no momentum)
                        v -= lr * grad
                    else:
                        # not trainable = BN stats = average latest value
                        v[...] = grad
                for item_id in item_ids_to_delete:
                    del kakiage_server.blobs[item_id]
        print()
        print("Running validation on server...")
        for k, v in model.state_dict().items():
            v[...] = torch.from_numpy(weights[snake2camel(k)])
        test_result = test(model, test_loader)
        print(test_result)
        test_results.append(test_result)
    torch.save(model.state_dict(), os.path.join(
        output_dir, "kakiage_trained_model.pt"))
    with open(os.path.join(output_dir, "kakiage_training.pkl"), "wb") as f:
        pickle.dump({"test_results": test_results}, f)
    onnx_path = os.path.join(
        output_dir, "kakiage_trained_model.onnx")
    print("exporting trained model with ONNX format: " + onnx_path)
    torch.onnx.export(model, torch.zeros(1, *input_shape), onnx_path,
                      input_names=["input"],
                      output_names=["output"])
    print("training ended. You can close the program by Ctrl-C.")
    # TODO: exit program (sys.exit(0) emits long error message)


asyncio.get_running_loop().create_task(main())
