import pyaudio
import websockets
import asyncio
import base64
import json
import subprocess
#import navigation 
#import speech_recognition as sr
#from main5 import menu_actions

#recognizer = sr.Recognizer()

auth_key = '360296f14f864e699942aee0a755bf27'
awake = False

FRAMES_PER_BUFFER = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

# starts recording
stream = p.open(
   format=FORMAT,
   channels=CHANNELS,
   rate=RATE,
   input=True,
   frames_per_buffer=FRAMES_PER_BUFFER
   #exception_on_overflow=False
)

# the AssemblyAI endpoint we're going to hit
URL = "wss://streaming.assemblyai.com/v3/ws"

async def send_receive():
    print(f"Connecting websocket to url {URL}")
    async with websockets.connect(
        URL,
        extra_headers={"Authorization": auth_key},
        ping_interval=5,
        ping_timeout=20,
    ) as ws:
     ack = await ws.recv()
     print("Connected:", ack)
     async def send(ws):
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    audio = base64.b64encode(data).decode("utf-8")
                    await ws.send(data)
                except websockets.exceptions.ConnectionClosedError as e:
                   #print(f"WebSocket closed: {e.code} – {e.reason}")
                    break
                await asyncio.sleep(0.01)


     async def receive(ws):
           global awake
           while True:
               try:              
                    result = await ws.recv()
                    allData = json.loads(result)
                    words = allData.get("words", [])
                    final_words = [w["text"] for w in words if w.get("word_is_final")]
                    if not final_words:
                      continue  

                    finalStr = " ".join(final_words).strip()
                    if finalStr == "":
                      continue

                    print("Result: ",finalStr)
                    if not awake and listen_for_wake_word(finalStr):
                      awake= True
                      print("Wake word detected")
               
                    elif awake and result.strip() != "":
                       runCommand(result)
                       awake = False

               except websockets.exceptions.ConnectionClosedError as e:
                   print(e)
                   print(f"WebSocket closed: {e.code} – {e.reason}")
                   break
               except Exception as e:
                print(e)
                break
     send_result, receive_result = await asyncio.gather(send(ws), receive(ws))





def listen_for_wake_word(text_2,wake_word="awake"):
    global awake
    if(wake_word in text_2):
        awake = True
        subprocess.call(["espeak", "Yes? I am listening."])
        return True
        
        


asyncio.run(send_receive())