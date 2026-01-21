Team ID: 22840-2
Overview:
The 2025-2026 Design Challenge for the TSA Software Development event is to develop a software program that removes barriers and increases accessibility for people with vision or hearing disabilities. The goal of our project is to create a web-based interface that accepts audio and/or video input, detect emotional tone or expression, and translate it into visual and/or spoken feedback for accessibility. This way, we aim to help people with either vision or hearing disabilities by translating emotional cues that normally would be missed.
System Components:
Input Devices:
-	Microphone (audio)
-	Webcam (video)
Processing Modules:
-	Audio processing - The model processes 2-second audio chunks at 16kHz sample rate and outputs emotion predictions
-	  SpeechBrain's inference
-	  Pre-trained emotion recognition model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP (huggingface link: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
-	  Wav2Vec2 as the underlying neural network architecture
-	  PyTorch and torchaudio for tensor operations and audio handling
-	  sounddevice for real-time audio stream input
-	  Dataset used for testing: Audio-Visual Database (RAVDESS)
-	Video Processing:
-	  Combined Pre-trained Emotion Detection Model (https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection)
-	    CNN trained on the FER2013 (Facial Expression Recognition 2013) dataset
-	    Possible emotions that can be classified: Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral
Libraries Used:
-	numpy – Mathematical operations, working with arrays.
-	sounddevice – For real-time microphone audio input.
-	speechbrain – Used for pretrained emotion recognition and inference.
-	torch – Used as a deep learning framework
-	torchaudio – Audio processing for PyTorch
-	tensorflow – Platform to deploy ML or AI models
-	Subprocess – Built-in python library that lets you spawn new processes
-	pyaudio -- Used for letting Python record/play audio
-	cv2 – For reading, displaying, and writing images
Current Progress:
So far, we’ve installed all the libraries and modules, and we’re currently testing each module to find the most effective one for our project. The next step would be to create a User Interface that implements these processing modules.
