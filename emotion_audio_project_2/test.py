import os
import warnings
import glob

# Disable torchcodec and suppress warnings
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import soundfile as sf
import numpy as np
import io
import pandas as pd
from scipy import signal
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


def load_audio_from_file(audio_path):
    """Load audio from a file using soundfile"""
    try:
        speech, sr = sf.read(audio_path)
        
        # Resample to 16000 Hz if needed
        if sr != 16000:
            num_samples = int(len(speech) * 16000 / sr)
            speech = signal.resample(speech, num_samples)
        
        # Convert to mono if stereo
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        
        return speech
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return np.array([])


def load_audio_from_bytes(audio_bytes):
    """Load audio from bytes using soundfile"""
    try:
        speech, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample to 16000 Hz if needed
        if sr != 16000:
            num_samples = int(len(speech) * 16000 / sr)
            speech = signal.resample(speech, num_samples)
        
        # Convert to mono if stereo
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        
        return speech
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.array([])


# Load audio files from Actor_01 folder
# Try to find the correct path
possible_paths = [
    "C:\\Users\\zixua\\OneDrive\\Desktop\\SoftwareDev\\software-development-2025\\emotion_audio_project\\audio_files_for_testing\\audio_speech_actors_01-24\\Actor_01",
    "C:\\Users\\zixua\\OneDrive\\Desktop\\SoftwareDev\\software-development-2025\\emotion_audio_project\\audio_files_for_testing\\Actor_01",
    "C:\\Users\\zixua\\OneDrive\\Desktop\\SoftwareDev\\audio_files_for_testing\\audio_speech_actors_01-24\\Actor_01",
]

audio_dir = None
for path in possible_paths:
    if os.path.exists(path):
        audio_dir = path
        print(f"Found audio directory: {audio_dir}")
        break

if audio_dir is None:
    print("Searching for Actor_01 directory...")
    for root, dirs, files in os.walk("C:\\Users\\zixua\\OneDrive\\Desktop\\SoftwareDev"):
        if "Actor_01" in dirs:
            audio_dir = os.path.join(root, "Actor_01")
            print(f"Found Actor_01 at: {audio_dir}")
            break

if audio_dir is None:
    print("ERROR: Could not find Actor_01 directory")
    exit()

# Find all wav files recursively
audio_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
print(f"Found {len(audio_files)} audio files in Actor_01")
audio_files = audio_files[:21]  # Limit to first 10 files for testing
# Process each audio file
processed_data = []
for audio_file in audio_files:
    speech = load_audio_from_file(audio_file)
    if speech.size > 0:  # Only add non-empty audio
        processed_data.append({
            "file": audio_file,
            "speech": speech
        })

print(f"Successfully loaded {len(processed_data)} audio files")

if len(processed_data) == 0:
    print("No audio files found. Exiting.")
    exit()

model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

# ============================================================================
# EMOTION RECOGNITION MODELS YOU CAN USE (all are fine-tuned for emotion):
# ============================================================================
# Option 1: Wav2Vec2 Large (MORE ACCURATE - recommended for better results)
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er")

# Option 2: HuBERT Base (GOOD alternative)
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

# Option 3: HuBERT Large (BEST accuracy - slower)
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

# Note: UniSpeech-SAT is not fine-tuned for emotion recognition, so it doesn't work
# ============================================================================

print("Loading model...")

# Optional: Load additional models for ensemble (uncomment to use)
# model_advanced = Wav2Vec2ForSequenceClassification.from_pretrained("superb/unispeech-sat-base-plus-superb-er")
# feature_extractor_advanced = Wav2Vec2FeatureExtractor.from_pretrained("superb/unispeech-sat-base-plus-superb-er")

# Get all audio samples
speech_list = [item["speech"] for item in processed_data]
file_list = [item["file"] for item in processed_data]

# Confidence threshold (0.0 to 1.0) - adjust this to filter predictions
CONFIDENCE_THRESHOLD = 0.5  # Change this value to be more or less strict

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(speech_list, sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits

# Get predictions with confidence scores
probabilities = torch.softmax(logits, dim=-1)
predicted_ids = torch.argmax(logits, dim=-1)
confidence_scores = torch.max(probabilities, dim=-1).values.detach().numpy()

labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

# Print results with confidence scores
print("\n" + "="*80)
print(f"Emotion Detection Results for Actor_01 (Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)")
print("="*80)
print(f"{'File Name':<40} {'Emotion':<15} {'Confidence':<10}")
print("-"*80)

high_confidence_count = 0
low_confidence_count = 0

for file_path, emotion, confidence in zip(file_list, labels, confidence_scores):
    file_name = os.path.basename(file_path)
    confidence_pct = confidence * 100
    
    # Mark low confidence predictions
    confidence_marker = ""
    if confidence < CONFIDENCE_THRESHOLD:
        confidence_marker = " ⚠️ LOW"
        low_confidence_count += 1
    else:
        high_confidence_count += 1
    
    print(f"{file_name:<40} {emotion:<15} {confidence_pct:>6.1f}%{confidence_marker}")

print("-"*80)
print(f"High Confidence (>{CONFIDENCE_THRESHOLD*100:.0f}%): {high_confidence_count} | Low Confidence: {low_confidence_count}")

# Print detailed probabilities for low-confidence predictions
print("\n" + "="*80)
print("Top Emotion Predictions with Probabilities (for review)")
print("="*80)

# Determine how many classes to show (max 3, but could be fewer)
num_classes = probabilities.shape[-1]
k = min(3, num_classes)

if k > 1:
    top_k_probs, top_k_indices = torch.topk(probabilities, k=k, dim=-1)
    
    for i, file_path in enumerate(file_list):
        if confidence_scores[i] < CONFIDENCE_THRESHOLD:
            file_name = os.path.basename(file_path)
            print(f"\n{file_name} (⚠️ LOW CONFIDENCE):")
            for prob, idx in zip(top_k_probs[i], top_k_indices[i]):
                emotion_name = model.config.id2label[idx.item()]
                print(f"  {emotion_name:<15} {prob.item()*100:>6.1f}%")