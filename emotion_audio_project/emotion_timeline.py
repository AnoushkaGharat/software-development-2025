# emotion_timeline.py
import audeer
import audonnx
import numpy as np
import librosa

# Load model (same as before)
model_root = 'model'
model = audonnx.load(model_root)

# Load audio
file_path = 'audio_files_for_testing/Actor_02/03-01-03-02-02-02-02.wav'
signal, sr = librosa.load(file_path, sr=16000)
signal = signal.astype(np.float32)

# Split into chunks
chunk_duration = 2  # seconds
chunk_size = sr * chunk_duration
timeline = []

for i in range(0, len(signal), chunk_size):
    chunk = signal[i:i+chunk_size]
    logits = model(chunk, sr)['logits'][0]  # arousal, dominance, valence
    t_start = i / sr
    t_end = min((i + chunk_size) / sr, len(signal)/sr)
    timeline.append((t_start, t_end, logits))

# Optional: map to simple emotion labels
def valence_to_emotion(valence, arousal):
    if valence > 0.6 and arousal > 0.5:
        return "Happy"
    elif valence < 0.4 and arousal > 0.5:
        return "Angry"
    elif valence < 0.4 and arousal < 0.5:
        return "Sad"
    else:
        return "Neutral"

for t in timeline:
    emotion_label = valence_to_emotion(t[2][2], t[2][0])
    print(f"{t[0]:.2f}-{t[1]:.2f} s: {emotion_label}")
