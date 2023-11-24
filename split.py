#'/path/to/your/mp3/files'
#'/Users/akansha_0501/Desktop/Violence Detection'

#training_directory = '/Users/akansha_0501/Desktop/Violence Detection/training'
#testing_directory = '/Users/akansha_0501/Desktop/Violence Detection/testing'

#In these 2 replace with 2 directories, where u want to store them


import os
import shutil
from random import shuffle

# Set the paths to your main folder containing two subfolders
main_directory = '/Users/akansha_0501/Desktop/Violence Detection/Dataset'
subfolder1 = 'Violence'  # Replace with the name of your first subfolder
subfolder2 = 'NonViolence'  # Replace with the name of your second subfolder

# Define the paths for training and testing directories
training_directory = '/Users/akansha_0501/Desktop/Violence Detection/training'
testing_directory = '/Users/akansha_0501/Desktop/Violence Detection/testing'

# Function to split data
def split_data(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    shuffle(files)
    split_index = int(len(files) * 0.8)
    training_set = files[:split_index]
    testing_set = files[split_index:]
    return training_set, testing_set

# Split data for subfolder 1
subfolder1_path = os.path.join(main_directory, subfolder1)
subfolder1_training, subfolder1_testing = split_data(subfolder1_path)

# Split data for subfolder 2
subfolder2_path = os.path.join(main_directory, subfolder2)
subfolder2_training, subfolder2_testing = split_data(subfolder2_path)

# Move files to respective training and testing directories
for file in subfolder1_training:
    src = os.path.join(subfolder1_path, file)
    dst = os.path.join(training_directory, file)
    shutil.move(src, dst)

for file in subfolder1_testing:
    src = os.path.join(subfolder1_path, file)
    dst = os.path.join(testing_directory, file)
    shutil.move(src, dst)

for file in subfolder2_training:
    src = os.path.join(subfolder2_path, file)
    dst = os.path.join(training_directory, file)
    shutil.move(src, dst)

for file in subfolder2_testing:
    src = os.path.join(subfolder2_path, file)
    dst = os.path.join(testing_directory, file)
    shutil.move(src, dst)