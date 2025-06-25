import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io

# --- Configuration Section ---
# Path to your service account key file
SERVICE_ACCOUNT_FILE = 'service_account.json'

# The ID of the folder to upload the file to.
# Get this from the folder's URL in your browser.
FOLDER_ID = "1-AQo-KnWjln2lkiUJEqgjlrF9sL3U9w7"

# The scopes define the level of access requested.
# For drive, 'https://www.googleapis.com/auth/drive' is a broad scope
# that allows for file creation, modification, etc.
SCOPES = ['https://www.googleapis.com/auth/drive']


def upload_with_overwrite(FILE_TO_UPLOAD : str):
    """Uploads a file, overwriting any existing file with the same name in the folder."""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # Extract the filename from the local path
    file_name = os.path.basename(FILE_TO_UPLOAD)

    try:
        # 1. Search for the file to see if it exists
        # ==========================================
        print(f"Searching for an existing file named '{file_name}'...")
        query = f"name = '{file_name}' and '{FOLDER_ID}' in parents and trashed = false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        media = MediaFileUpload(FILE_TO_UPLOAD, resumable=True)

        if files:
            # 2a. If file exists, update it
            # ==============================
            existing_file = files[0]
            file_id = existing_file['id']
            print(f"File found with ID: {file_id}. Updating content...")

            updated_file = service.files().update(
                fileId=file_id,
                media_body=media,
                fields='id, name'
            ).execute()
            print(f"File '{updated_file.get('name')}' was successfully updated.")

        else:
            # 2b. If file does not exist, create it
            # =====================================
            print("File not found in the drive. Uploading...")
            file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}

            created_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            print(f"File '{created_file.get('name')}' was successfully created with ID: {created_file.get('id')}")

    except HttpError as error:
        print(f'An error occurred: {error}')
    except FileNotFoundError:
        print(f"Error: The local file '{FILE_TO_UPLOAD}' was not found.")


def download_file_from_folder(FILE_NAME_TO_DOWNLOAD : str, LOCAL_DOWNLOAD_PATH :str = "context.txt"):
    """Searches for a file by name and downloads it, overwriting any local file."""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    try:
        # Search for the file
        query = f"name = '{FILE_NAME_TO_DOWNLOAD}' and '{FOLDER_ID}' in parents and trashed = false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        if not files:
            print(f"Error: No file named '{FILE_NAME_TO_DOWNLOAD}' found in the folder.")
            return

        file_id = files[0].get('id')
        file_name = files[0].get('name')

        # Download the file
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        # This 'wb' mode will automatically overwrite the local file if it exists.
        with open(LOCAL_DOWNLOAD_PATH, 'wb') as f:
            f.write(fh.getvalue())

        print(f"File '{file_name}' downloaded and overwritten successfully to '{LOCAL_DOWNLOAD_PATH}'.")

    except HttpError as error:
        print(f'An error occurred: {error}')

if __name__ == '__main__':

    file_to_upload = "templates/pouring_cram_updated.html"

    upload_with_overwrite(file_to_upload)
    #
    #
    # file_name_to_download = "pouring_cram_updated.html"
    # local_download_path = "pouring_static.html"
    # download_file_from_folder(file_name_to_download, local_download_path)