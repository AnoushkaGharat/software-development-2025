import sounddevice as sd
import numpy as np
import torch
import torchaudio
# Monkey patch for compatibility with older SpeechBrain versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: [] 

from speechbrain.inference.interfaces import foreign_class
from speechbrain.inference.interfaces import foreign_class

# 1. Load the model
print("Loading model... please wait.")
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
    pymodule_file="custom_interface.py", 
    classname="CustomEncoderWav2vec2Classifier"
)

# Configuration
SAMPLE_RATE = 16000
WINDOW_SEC = 2  # Analyzes audio in 2-second chunks

def callback(indata, frames, time, status):
    if status:
        print(f"Error in stream: {status}")
    
    # 2. Convert NumPy array to Torch Tensor
    # SpeechBrain expects a tensor with shape [batch, time]
    signal = torch.from_numpy(indata.copy()).float().squeeze().unsqueeze(0)
    
    # 3. Perform Inference
    # .classify_batch returns: (out_prob, score, index, text_lab)
    with torch.no_grad():
        _, _, _, text_lab = classifier.classify_batch(signal)
    
    print(f"Predicted Emotion: {text_lab[0]}")

# 4. Start the microphone stream
print(f"Listening... (Analyzing every {WINDOW_SEC} seconds)")
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=int(SAMPLE_RATE * WINDOW_SEC)
    ):
        while True:
            sd.sleep(1000)  # Keep the main thread alive
except KeyboardInterrupt:
    print("\nStopped by user.")

# Anindita's aliveeeee