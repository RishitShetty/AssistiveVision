import asyncio
import websockets
import json
import socket
import pyttsx3
import time

# Initialize pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 250)  # Adjust speech rate
engine.setProperty('volume', 1.0)  # Max volume

# List of possible server IPs
SERVER_URIS = [
    #"ws://172.30.55.158:8765/video",
    "ws://172.20.10.4:8765/video",
    "ws://192.168.137.142:8765/command",
    #"ws://172.30.44.117:8765/video",
    #"ws://172.30.32.1:8765/video",
    "ws://192.168.137.1:8765/command"
]

# Last time an alert was spoken
last_alert_time = 0

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

def speak_alert(message):
    """Function to use pyttsx3 for speech output with a 5-second delay"""
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time >= 5:  # Ensure a 5-second delay between alerts
        engine.say(message)
        engine.runAndWait()
        last_alert_time = current_time

async def activate_obstacle_detection(uri):
    async with websockets.connect(uri) as websocket:
        command = {"command": "start_obstacle_detection"}
        await websocket.send(json.dumps(command))
        print("Sent start_obstacle_detection command; now receiving detection updates...")
        try:
            while True:
                response = await websocket.recv()
                #print("Detection Update:", response)

                # Parse response (assuming it's in JSON format)
                try:
                    data = json.loads(response)
                    if "objects" in data and isinstance(data["objects"],list) and len(data["objects"])>0:
                        proximity = data["objects"][0].get("proximity")
                        print(f"Parsed proximity: {proximity}")  # Debugging

                        if proximity == "VERY CLOSE":
                            speak_alert("Alert! Potential obstacle is very close.")
                        elif proximity == "NEARBY":
                            speak_alert("Alert! Potential obstacle is nearby.")
                        else:
                            print(f"Unexpected proximity value: {proximity}")  # Handle unknown values
                    else:
                        print("NF")

                except json.JSONDecodeError:
                    print("Received invalid JSON data:", response)

        except websockets.ConnectionClosed:
            print("Connection closed by server.")

if __name__ == '__main__':
    # Automatically select the working server
    selected_uri = None
    for uri in SERVER_URIS:
        if check_server(uri):
            selected_uri = uri
            break

    if selected_uri:
        print(f"Using server: {selected_uri}")
        asyncio.run(activate_obstacle_detection(selected_uri))
    else:
        print("No available servers found. Please check your connections.")
