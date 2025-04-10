import os

def rename_files(folder_path):
    # Get a sorted list of all .png files in the folder
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    
    counter = 86

    for file in files:
        
        # Create new file name with zero-padded numbers
        new_name = f"axial_{counter:03d}.png"
        
        # Get full paths
        old_file = os.path.join(folder_path, file)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {file} -> {new_name}")
        
        # Increment counter
        counter += 1

# Replace with the path to your folder
folder_path = "/Users/liumengyuan/Downloads/muscle_data/06upper"  # Change this to your folder path
rename_files(folder_path)
