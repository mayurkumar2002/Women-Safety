import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
# from PIL import Image
import librosa
from tensorflow.image import resize
from werkzeug.utils import secure_filename
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# import soundfile as sf
# import sounddevice as sd



from flask import Flask, request, jsonify

model = keras.models.load_model("audio_classification_model.h5")

target_shape = (256, 256)

classes = ['Danger', 'Normal']

# Function to preprocess and classify an audio file
def test_audio(file_path):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Load the audio file.
    # audio_data, sample_rate = sf.read(file_path,sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    predictions = model.predict(mel_spectrogram)

    class_probabilities = predictions[0]

    predicted_class_index = np.argmax(class_probabilities)
    predicted_class = classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]*100


    return predicted_class, accuracy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(
    app.instance_path, 
    'uploads'
)
try: 
    os.makedirs(app.config['UPLOAD_FOLDER'])
except: 
    pass 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('audio')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            dest = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                secure_filename(file.filename)
            )
            # Save the file on the server.
            file.save(dest)
            predicted_class, accuracy = test_audio(dest)
            os.remove(dest)
            data = {"predicted_class": predicted_class, "accuracy": int(accuracy)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)