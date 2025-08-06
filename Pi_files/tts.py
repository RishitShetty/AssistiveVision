import asyncio
import websockets
import pyttsx3
import socket


# List of possible server IPs
SERVER_URIS = [
    #"ws://172.30.55.158:8765/video",
    #"ws://172.20.10.4:8765/video",
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


async def send_command(uri, command):
    async with websockets.connect(uri) as websocket:
        print("Connected to command server.")
        await websocket.send(command)
        print("Sent command:", command)
        response = await websocket.recv()
        print("Received response:", response)

        # Use text-to-speech to announce the location
        engine = pyttsx3.init()
        engine.say(f"Your location is {response}")
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
        asyncio.run(send_command(uri, command))
    else:
        print("No available servers found. Please check your connections.")


    
