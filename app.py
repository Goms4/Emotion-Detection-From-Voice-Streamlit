import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
tf.compat.v1.reset_default_graph()
from sklearn.preprocessing import LabelEncoder
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.keras')

# Load the LabelEncoder classes
le = LabelEncoder()
le.classes_ = np.load('classes.npy')

# Mapping of integer labels to emotion names based on provided info
emotion_map = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprise'
}

# Function to extract features from an audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# Function to detect gender based on the filename convention
def detect_gender(file_path):
    filename = os.path.basename(file_path)
    try:
        actor = int(filename.split('-')[-1].split('.')[0])
        return 'female' if actor % 2 == 0 else 'male'
    except ValueError:
        return 'unknown'  # Default to unknown if filename does not match expected format

# Placeholder for language detection function
def detect_language(file_path):
    # Placeholder implementation, needs proper language detection
    return 'english'  # Default to 'english' for now

# Streamlit app setup
st.title('Emotion Detection from Voice')

# Choice for upload or record
option = st.radio("Choose an option", ("Upload an audio file", "Record audio"))

if option == "Upload an audio file":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = "temp_audio_file.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check for gender
        gender = detect_gender(temp_file_path)
        if gender != 'female' and gender != 'unknown':
            st.write('Please upload a female voice.')
        # Check for language
        elif detect_language(temp_file_path) != 'english':
            st.write('Please upload an audio file in English.')
        else:
            # Extract features and predict emotion
            features = extract_features(temp_file_path)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)  # Reshape for LSTM model
            prediction = model.predict(features)
            predicted_emotion_idx = np.argmax(prediction)
            predicted_emotion = emotion_map.get(predicted_emotion_idx + 1, 'Unknown')
            st.write(f'Predicted Emotion: {predicted_emotion}')

            # Remove the temporary file
            os.remove(temp_file_path)
else:
    # Use streamlit-webrtc for recording audio
    def audio_recorder_callback(frame):
        audio_data = frame.to_ndarray().flatten()
        temp_file_path = "temp_audio_recording.wav"
        librosa.output.write_wav(temp_file_path, audio_data, sr=44100)
        return temp_file_path

    webrtc_ctx = webrtc_streamer(
        key="key",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            frame = audio_frames[-1]
            temp_file_path = audio_recorder_callback(frame)

            # Check for gender
            gender = detect_gender(temp_file_path)
            if gender != 'female' and gender != 'unknown':
                st.write('Please record a female voice.')
            # Check for language
            elif detect_language(temp_file_path) != 'english':
                st.write('Please record audio in English.')
            else:
                # Extract features and predict emotion
                features = extract_features(temp_file_path)
                features = np.expand_dims(features, axis=0)
                features = np.expand_dims(features, axis=2)  # Reshape for LSTM model
                prediction = model.predict(features)
                predicted_emotion_idx = np.argmax(prediction)
                predicted_emotion = emotion_map.get(predicted_emotion_idx + 1, 'Unknown')
                st.write(f'Predicted Emotion: {predicted_emotion}')

                # Remove the temporary file
                os.remove(temp_file_path)
