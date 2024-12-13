import os
import shutil

def clear_directory(directory):
    """Remove all files and folders in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory does not exist, creating: {directory}")
        os.makedirs(directory)
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted folder: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def copy_files(source, destination):
    """Copy all files from source to destination."""
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"Copied files from {source} to {destination}")
    except Exception as e:
        print(f"Failed to copy from {source} to {destination}. Reason: {e}")

def main():
    source_folder = r"E:\mcmodding\fabric-example-mod\instances\client1"
    destination_folders = [
        rf"E:\mcmodding\fabric-example-mod\instances\client{i}" for i in range(2, 16)
    ]

    for dest in destination_folders:
        print(f"\nProcessing destination: {dest}")
        clear_directory(dest)
        copy_files(source_folder, dest)

if __name__ == "__main__":
    main()