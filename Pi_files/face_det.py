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


async def face_detection(uri):
    async with websockets.connect(uri) as websocket:
        start_cmd = {"command": "start_face_det", "duration": 10}
        await websocket.send(json.dumps(start_cmd))
        print("Sent start_face_det command; waiting for recognition result...")
        
        response = await websocket.recv()
        print("Server response:", response)
        
        # Parse JSON response
        try:
            result = json.loads(response)
            status = result.get("status")

            engine = pyttsx3.init()

            if status == "recognized":
                name = result.get("name", "Unknown Person")
                print(f"Recognized person: {name}")
                engine.say(f"The recognized person is {name}")
            
            elif status == "not_recognized":
                print("Face detected but not recognized.")
                engine.say("Face detected but not recognized.")
            
            elif status == "no_face_detected":
                print("No face detected.")
                engine.say("No face detected.")
            
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
        asyncio.run(face_detection(selected_uri))
    else:
        print("No available servers found. Please check your connections.")



