import os
import zipfile


def extract_zip(zip_path: str, destination_path: str) -> None:
    """
    Extracts a ZIP file to the specified destination path.

    :param zip_path: Path to the ZIP file.
    :param destination_path: Destination path where the ZIP file will be extracted.
    """
    # Check if the ZIP file exists
    if not os.path.exists(zip_path):
        print(f"ZIP file '{zip_path}' does not exist.")
        return

    # Create the destination path if it does not exist
    os.makedirs(destination_path, exist_ok=True)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

    print(f"ZIP file has been extracted to: {destination_path}")


if __name__ == "__main__":
    extract_zip("data/FER-2013.zip", "data")
