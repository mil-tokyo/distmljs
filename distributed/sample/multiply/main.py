import asyncio
from typing import List
from uuid import uuid4
import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from kakiage.server import KakiageServerWSConnectEvent, KakiageServerWSReceiveEvent, setup_server
from kakiage.tensor_serializer import serialize_tensors_to_bytes, deserialize_tensor_from_bytes

kakiage_server = setup_server()
app = kakiage_server.app


async def main():
    print("Multiply sample")
    client_ids = []
    print("Waiting a client to connect")
    while True:
        event = await kakiage_server.event_queue.get()
        if isinstance(event, KakiageServerWSConnectEvent):
            client_ids.append(event.client_id)
            break
        else:
            print("unexpected event")
    src_array = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    print("computing [1.0,1.5,2.0] * 2.0 in client")
    src_s = serialize_tensors_to_bytes({"src": src_array})
    src_item_id = uuid4().hex
    dst_item_id = uuid4().hex
    kakiage_server.blobs[src_item_id] = src_s
    await kakiage_server.send_message(client_ids[0], {"src": src_item_id, "dst": dst_item_id})
    while True:
        event = await kakiage_server.event_queue.get()
        if isinstance(event, KakiageServerWSReceiveEvent):
            break
        else:
            print("unexpected event")
    dst_s = kakiage_server.blobs[dst_item_id]
    dst_array = deserialize_tensor_from_bytes(dst_s)
    print(dst_array)


asyncio.get_running_loop().create_task(main())
