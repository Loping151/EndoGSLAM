import os
import shutil
import argparse

def classify_files_in_folder(dataset_folder_path):
    """
    Classifies files in the specified folder by their types, extracted from the file names.
    Each type is moved into a subfolder named after the type.
    Files are expected to have names in the format: "number_type.extension".
    """
    for filename in os.listdir(dataset_folder_path):
        # Ensure the filename starts with a digit and contains "_"
        if filename[0].isdigit() and "_" in filename:
            type_name = filename.split("_")[-1].split(".")[0]
            type_folder_path = os.path.join(dataset_folder_path, type_name)
            if not os.path.exists(type_folder_path):
                os.makedirs(type_folder_path)
            
            original_file_path = os.path.join(dataset_folder_path, filename)
            new_file_path = os.path.join(type_folder_path, filename)
            shutil.move(original_file_path, new_file_path)
    print(f"Files classified in: {dataset_folder_path}")

def process_folders(root_folder):
    """
    Processes each subfolder within the root folder, applying file classification.
    Skips folders that do not contain files starting with a digit, assuming they've been processed.
    """
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            has_digit_starting_files = any(f[0].isdigit() for f in os.listdir(folder_path))
            if has_digit_starting_files:
                classify_files_in_folder(folder_path)
            else:
                print(f"Skipping already processed folder: {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify files in dataset folders based on their types.")
    parser.add_argument("--dataset_path", default="./data/C3VD", help="Path to the dataset root folder. Default is './data'.")
    args = parser.parse_args()

    process_folders(args.dataset_path)
