import asyncio
import websockets
import json
import pyttsx3
import socket


# List of possible server IPs
SERVER_URIS = [
    #"ws://172.30.55.158:8765/video",
    "ws://172.20.10.4:8765/video",
    "ws://192.168.137.142:8765/command",
    #"ws://172.30.44.117:8765/video",
    #"ws://172.30.32.1:8765/video",
    "ws://192.168.137.1:8765/command"
]

def check_server(ip):
    """Check if a server is reachable on port 8765"""
    host = ip.replace("ws://", "").split(":")[0]  # Extract just the IP
    port = 8765  # WebSocket server port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # 1-second timeout
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"✅ Connection successful: {host}:{port}")
        else:
            print(f"❌ Connection failed: {host}:{port}")
        return result == 0  # Returns True if server is reachable

async def start_ocr(uri):
    async with websockets.connect(uri) as websocket:
        # Send command to start OCR
        start_cmd = {"command": "start_ocr"}
        await websocket.send(json.dumps(start_cmd))
        print("Sent start_ocr command; waiting for OCR result...")

        # Receive OCR result from server
        response = await websocket.recv()
        print("Server response:", response)

        # Parse JSON response
        try:
            result = json.loads(response)
            status = result.get("closest_match")

            engine = pyttsx3.init()

            if status:
                print(f"Detected text: {status}")
                engine.say(f"Detected text is {status}")

            elif status == "no_text":
                print("No text detected.")
                engine.say("No text detected.")

            else:
                print("Received unknown status from server.")
                engine.say("Received unknown status from server.")

            engine.runAndWait()

        except json.JSONDecodeError:
            print("Failed to decode server response.")
            engine = pyttsx3.init()
            engine.say("Failed to decode server response.")
            engine.runAndWait()



if __name__ == '__main__':
    # Automatically select the working server
    selected_uri = None
    for uri in SERVER_URIS:
        if check_server(uri):
            selected_uri = uri
            break

    if selected_uri:
        print(f"Using server: {selected_uri}")
        asyncio.run(start_ocr(uri))
    else:
        print("No available servers found. Please check your connections.")


