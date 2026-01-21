import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    print("Trying to load UniSpeech-SAT model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/unispeech-sat-base-plus-superb-er")
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading UniSpeech-SAT: {e}")
    print("\nTrying Wav2Vec2 Base instead...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
    print("✓ Wav2Vec2 Base loaded successfully!")
