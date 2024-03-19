import asyncio
import websockets

async def run_client (websocket):
    while True:
        try:
            message = await websockets.recv()
        except websockets.ConnectionClosedOK:
            break
        pring(message)

async def main ():
    async with websockets.serve(handler, "", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
