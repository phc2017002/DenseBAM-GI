import os
import subprocess

# Directory containing the images
directory = './test_HMER/'

# List all files in the directory
files = os.listdir(directory)

# Sort the files to ensure consistent order
files.sort()

# Path to your Python file
python_file = './Densenet_testway_BAM_GI.py'

# Loop through each file and call your Python file with the file as an argument
for file_name in files:
    # Construct the full path to the file
    file_path = os.path.join(directory, file_name)
    
    # Check if the file is an image (you can modify this condition based on your file types)
    if file_name.endswith('.bmp') or file_name.endswith('.png') or file_name.endswith('.jpg'):
        # Call your Python file with the file as an argument using subprocess
        subprocess.run(['python', python_file, file_path])
        #print(output.decode())
