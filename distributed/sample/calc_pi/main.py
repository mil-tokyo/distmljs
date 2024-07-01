import asyncio
from typing import List
import numpy as np
from distmljs.server import (
    DistMLJSServerWSConnectEvent,
    DistMLJSServerWSReceiveEvent,
    DistMLJSServerWSDisconnectEvent,
    setup_server,
)

distmljs_server = setup_server()
app = distmljs_server.app


def update_status(n_clients, n_points, computed_pi):
    print(
        f"\rPI={computed_pi:.16f}, points={n_points}, workers={n_clients}                      ",
        end="",
    )


async def main():
    print("Montecarlo PI computing sample")
    client_ids = []
    run_per_ticket = 10000000
    run_total = 0
    hit_total = 0
    print("Waiting a client to connect")
    while True:
        event = await distmljs_server.event_queue.get()
        if isinstance(event, DistMLJSServerWSConnectEvent):
            client_ids.append(event.client_id)
            await distmljs_server.send_message(
                event.client_id, {"run_per_ticket": run_per_ticket}
            )
        elif isinstance(event, DistMLJSServerWSDisconnectEvent):
            client_ids.remove(event.client_id)
            print("DISCONNECTED")
        elif isinstance(event, DistMLJSServerWSReceiveEvent):
            run_total += event.message["run"]
            hit_total += event.message["hit"]
            computed_pi = hit_total * 4 / run_total
            update_status(len(client_ids), run_total, computed_pi)
            try:
                await distmljs_server.send_message(
                    event.client_id, {"run_per_ticket": run_per_ticket}
                )
            except:
                # occurs when disconnected just after worker sends result
                pass
        else:
            print("unexpected event")
            break


asyncio.get_running_loop().create_task(main())
