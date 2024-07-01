import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse


@dataclass
class DistMLJSServerEvent:
    pass


@dataclass
class DistMLJSServerWSEvent(DistMLJSServerEvent):
    client_id: str


@dataclass
class DistMLJSServerWSConnectEvent(DistMLJSServerWSEvent):
    pass


@dataclass
class DistMLJSServerWSReceiveEvent(DistMLJSServerWSEvent):
    message: object


@dataclass
class DistMLJSServerWSDisconnectEvent(DistMLJSServerWSEvent):
    pass


app = FastAPI()
blobs = {}
event_queue: "asyncio.Queue[DistMLJSServerEvent]" = asyncio.Queue()
ws_clients: Dict[str, WebSocket] = {}


class DistMLJSServer:
    app: FastAPI
    blobs: Dict[str, bytes]
    event_queue: "asyncio.Queue[DistMLJSServerEvent]"

    def __init__(
        self,
        app: FastAPI,
        blobs: Dict[str, bytes],
        event_queue: "asyncio.Queue[DistMLJSServerEvent]",
    ) -> None:
        self.app = app
        self.blobs = blobs
        self.event_queue = event_queue

    async def send_message(self, client_id: str, message: object) -> None:
        client = ws_clients.get(client_id)
        if client is None:
            raise KeyError("client does not exist")
        await client.send_json(message)


server = DistMLJSServer(app, blobs, event_queue)


# avoid cache to force update js file when reloaded
@app.middleware("http")
async def add_my_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store"
    return response


@app.put("/distmljs/blob/{item_id}")
async def binary_put(item_id: str, request: Request):
    raw_data = await request.body()  # bytes
    blobs[item_id] = raw_data
    return {"item_id": item_id, "length": len(raw_data)}


@app.get("/distmljs/blob/{item_id}")
async def binary_get(item_id: str):
    item = blobs.get(item_id)
    if item is not None:
        return Response(content=item, media_type="application/octet-stream")
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.websocket("/distmljs/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = uuid4().hex
    ws_clients[client_id] = websocket
    await event_queue.put(DistMLJSServerWSConnectEvent(client_id))
    try:
        while True:
            data = await websocket.receive_json()
            await event_queue.put(DistMLJSServerWSReceiveEvent(client_id, data))
    except WebSocketDisconnect:
        del ws_clients[client_id]
        await event_queue.put(DistMLJSServerWSDisconnectEvent(client_id))


def setup_server(default_static=True) -> DistMLJSServer:
    # 今はグローバルオブジェクトを返すだけだが、将来的にはグローバルな状態を持たないようにする

    # staticファイルの配信設定
    # / で public/index.html
    # /static/* で public/static/*
    # を配信
    if default_static:
        app.mount("/static", StaticFiles(directory="public/static"), name="static")

        @app.get("/")
        async def read_index():
            return FileResponse("public/index.html")

    return server
