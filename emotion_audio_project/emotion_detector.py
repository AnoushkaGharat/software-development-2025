# emotion_detector.py
import audeer
import audonnx
import numpy as np
import librosa

# Download and extract model
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)

# Load model
model = audonnx.load(model_root)

# Load your audio
file_path = 'audio_files_for_testing/Actor_02/03-01-03-02-02-02-02.wav'  # <-- put your audio in the project folder
signal, sr = librosa.load(file_path, sr=16000)
signal = signal.astype(np.float32)

# Run model
output = model(signal, sr)
print(output['logits'])
