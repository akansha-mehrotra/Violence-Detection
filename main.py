import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.applications import VGG16
import pickle

# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        print(f"Retrieving frame - ret: {ret}")
        if not ret:
            print("Video reading finished or failed")
            break
        print("Frame shape:", frame.shape)  # Check the frame shape
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # VGG model expects RGB format
        frames.append(frame)
        print(frame.shape)
    cap.release()
    frames = np.array(frames)
    print("Number of frames read:", len(frames))  # Check the number of frames read
    features = base_model.predict(frames)
    return features


# Paths to directories
train_directory = ''
test_directory = ''


# Function to extract features and labels for individual video files
def extract_data_from_directory(directory):
    features = []
    labels = []
    max_frames = 0  
    print("Feature Extraction")
    for video_file in os.listdir(directory):
        video_path = os.path.join(directory, video_file)

        if not video_file.endswith(('.mp4', '.avi', '.mov')):
            print("Skipping non-video file:", video_file)
            continue

        extracted_features = extract_features_from_video(video_path)
        if extracted_features.shape[0] > 0:  # Checking if features were extracted
            print("Features Extracted")
            features.append(extracted_features)
            labels.append(1 if 'Violence' in directory else 0)  # Assigning labels based on directory name
            # Track the maximum number of frames
            print("Tracking max frames")
            max_frames = max(max_frames, extracted_features.shape[0])
    
    # Padding the features to have a consistent number of frames (if necessary)
    padded_features = []
    for feature in features:
        pad_length = max_frames - feature.shape[0]
        if pad_length > 0:
            print("Padding features")
            feature = np.pad(feature, [(0, pad_length), (0, 0), (0, 0), (0, 0)], mode='constant')
        padded_features.append(feature)
    
    features = np.array(padded_features)
    labels = np.array(labels)
    
    # Reshaping features for LSTM
    features_lstm = np.reshape(features, (features.shape[0], features.shape[1], -1))

    return features, labels, features_lstm


def check_frame_shape(frame):
    if frame is not None:
        print(f"Frame shape: {frame.shape}")
    else:
        print("Frame is None")


def check_shapes(train_features_flat, train_labels, test_features_flat, test_labels):
    print("Train Features shape:", train_features_flat.shape)
    print("Train Labels shape:", train_labels.shape)
    print("Test Features shape:", test_features_flat.shape)
    print("Test Labels shape:", test_labels.shape)

# Loading data and extract features including features for LSTM
train_features, train_labels, train_features_lstm = extract_data_from_directory(train_directory)
test_features, test_labels, test_features_lstm = extract_data_from_directory(test_directory)

# Flattening and preparing features for a Dense Neural Network model 
train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

check_shapes(train_features_flat, train_labels, test_features_flat, test_labels)


def create_dense_neural_network(train_features_flat, train_labels, test_features_flat, test_labels):
    input_shape = train_features_flat.shape[1:]

    # Creating a Sequential model
    print("Creating Dense Neural Network")
    dense_model = Sequential()

    # Adding layers to the model
    dense_model.add(Dense(256, activation='relu', input_shape=input_shape))
    dense_model.add(Dropout(0.5))
    dense_model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    # Compiling the model
    print("Compiling the DNN Model")
    dense_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    print("Training DNN Model")
    dense_model.fit(train_features_flat, train_labels, epochs=10, batch_size=32, validation_data=(test_features_flat, test_labels))
    print("DNN Model training complete")

    return dense_model



def create_lstm_model(train_features_lstm, train_labels, test_features_lstm, test_labels):
    # Creating an LSTM model
    print("Creating LSTM Model")
    lstm_model = Sequential()

    # Adding LSTM and Dense layers to the model
    lstm_model.add(LSTM(128, input_shape=(train_features_lstm.shape[1], train_features_lstm.shape[2])))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    # Compiling the model
    print("Compiling the LSTM model")
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    print("Training LSTM Model")
    lstm_model.fit(train_features_lstm, train_labels, epochs=10, batch_size=32, validation_data=(test_features_lstm, test_labels))
    print("Done training LSTM Model")

    return lstm_model

# Create and train Dense Neural Network model
dense_model = create_dense_neural_network(train_features_flat, train_labels, test_features_flat, test_labels)
with open('dense_model.pkl', 'wb') as dense_file:
    pickle.dump(dense_model, dense_file)

# Create and train LSTM model
lstm_model = create_lstm_model(train_features_lstm, train_labels, test_features_lstm, test_labels)
with open('lstm_model.pkl', 'wb') as lstm_file:
    pickle.dump(lstm_model, lstm_file)

# Loading the models using pickle
with open('dense_model.pkl', 'rb') as dense_file:
    loaded_dense_model = pickle.load(dense_file)

with open('lstm_model.pkl', 'rb') as lstm_file:
    loaded_lstm_model = pickle.load(lstm_file)

#Function for prediction
def predict_violence_in_random_clip(random_clip_path, model):
    extracted_features = extract_features_from_video(random_clip_path)
    
    if extracted_features.shape[0] > 0:
        extracted_features_lstm = np.reshape(extracted_features, (1, extracted_features.shape[0], -1))
        
        prediction = model.predict(extracted_features_lstm)
        prediction_value = prediction[0][0]
        
        if prediction_value >= 0.5:
            return "Violent", prediction_value
        else:
            return "Not Violent", prediction_value
    else:
        return "Features could not be extracted from the video.", 0.0

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Example usage for predicting violence in a random clip using loaded models
random_clip_path = '' # Enter path to clip
prediction_result_dnn, pvalue_dnn = predict_violence_in_random_clip(random_clip_path, loaded_dense_model)
print("Prediction for the random clip (Dense NN):", prediction_result_dnn)

prediction_result_lstm, pvalue_lstm = predict_violence_in_random_clip(random_clip_path, loaded_lstm_model)
print("Prediction for the random clip (LSTM):", prediction_result_lstm)

# Calculate RMSE using the loaded models
rmse_dnn = calculate_rmse(test_labels, pvalue_dnn)
print("Root Mean Square Error (Dense Neural Network):", rmse_dnn)

rmse_lstm = calculate_rmse(test_labels, pvalue_lstm)
print("Root Mean Square Error (LSTM):", rmse_lstm)
