import os
import re

def rename_files(folder_path):

    files = os.listdir(folder_path)

    pattern = re.compile(r'^(\d+)_color\.png$')

    for filename in files:
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            new_number = number.zfill(4)
            new_filename = f"{new_number}_color.png"
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f"Renamed {filename} to {new_filename}")

folder_path = 'cecum_t1_a/color'
rename_files(folder_path)
