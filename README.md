# Emotion-Detection-From-Voice-Streamlit
1. **Dataset Exploration:** I used the RAVDESS dataset, which contains audio recordings of actors speaking various sentences with different emotions. Each audio file is labeled with the emotion expressed and other metadata like actor gender.

2. **Data Preprocessing:** I extracted features from the audio files using librosa, a Python library for audio and music analysis. The main feature extracted was the Mel-frequency cepstral coefficients (MFCCs), which represent the short-term power spectrum of a sound.

3. **Model Training:** I built an LSTM (Long Short-Term Memory) neural network model using Keras to classify emotions based on the extracted audio features. The model was trained on the preprocessed dataset, with emotions encoded as numerical labels.

4. **Streamlit App Development:** I created a Streamlit web application to allow users to upload or record audio files and predict the emotion expressed in the audio. For audio recording, I used the `streamlit-webrtc` library, which enables real-time audio input from the browser.

5. **Integration and Deployment:** I integrated the trained model into the Streamlit app and deployed it locally. Users can interact with the app through a simple interface, either uploading pre-recorded audio files or recording their voice directly in the browser. The app then predicts the emotion expressed in the audio and displays the result to the user.

Overall, this project combines data preprocessing, machine learning model development, and web application development to create an interactive tool for emotion detection from voice recordings.
