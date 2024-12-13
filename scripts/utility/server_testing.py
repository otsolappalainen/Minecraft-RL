import asyncio
import websockets
import json
import time

async def send_turn_right(uri):
    message = {
        "action": "reset 2",
    }

    async with websockets.connect(uri) as websocket:
        # Send the message
        await websocket.send(json.dumps(message))
        print(f"Sent to {uri}: {message}")

        # Receive the response (if any)
        response = await websocket.recv()
        #print(f"Received from {uri}: {response}")

async def main():
    uris = [
        "ws://localhost:8080",
        "ws://localhost:8081",
        "ws://localhost:8082",
        "ws://localhost:8083",
        "ws://localhost:8084",
        "ws://localhost:8085",
        "ws://localhost:8086",
        "ws://localhost:8087",
        "ws://localhost:8088",
    ]

    for uri in uris:
        await send_turn_right(uri)
        await asyncio.sleep(1)  # Delay of 2 seconds between each message

# Run the async function
asyncio.get_event_loop().run_until_complete(main())


#/give @a minecraft:torch 300