import requests
import argparse
import os


def get_best_server():
    """Gets the best available GoFile server."""
    try:
        response = requests.get("https://api.gofile.io/servers")
        response.raise_for_status()  # Raise an exception for bad status codes
        servers = response.json()["data"]["servers"]
        # The first server in the list is generally the best one
        return servers[0]["name"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting server: {e}")
        return None
    except (KeyError, IndexError):
        print("Error: Could not parse the server list from GoFile.")
        return None


def upload_file(file_path):
    """Uploads a file to GoFile."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    server = get_best_server()
    if not server:
        return

    print(f"Uploading '{os.path.basename(file_path)}' to {server}...")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(f"https://{server}.gofile.io/contents/uploadfile", files=files)
            response.raise_for_status()

            response_data = response.json()

            # Check if the status is 'ok' before trying to parse 'data'
            if response_data.get("status") == "ok":
                data = response_data.get("data")
                if data:
                    print("\n--- Upload Successful! ---")
                    # Use .get() for all keys to prevent KeyErrors and provide defaults
                    print(f"File Name: {data.get('fileName', 'N/A')}")
                    print(f"Download Page: {data.get('downloadPage', 'N/A')}")
                    print(f"Direct Download Link: {data.get('directLink', 'N/A')}")
                    print(f"Admin Code (for managing file): {data.get('adminCode', 'N/A')}")
                    print("--------------------------")
                else:
                    print("Error: 'data' field not found in the successful server response.")
                    print("--- Server Response ---")
                    print(response.text)
                    print("-----------------------")
            else:
                print("Error: The upload was not successful according to the server.")
                print("--- Server Response ---")
                print(response.text)
                print("-----------------------")

    except requests.exceptions.RequestException as e:
        print(f"Error during upload request: {e}")
    except requests.exceptions.JSONDecodeError:
        print("Error: Failed to decode server response as JSON.")
        print("--- Server Response ---")
        print(response.text)
        print("-----------------------")


def download_single_file(url, name, save_path):
    """Helper function to download a single file from a direct URL."""
    print(f"Downloading '{name}'...")
    try:
        # It's good practice to create the save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            file_save_path = os.path.join(save_path, name)
            with open(file_save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"  -> Successfully saved to: {os.path.abspath(file_save_path)}")
    except requests.exceptions.RequestException as e:
        print(f"  -> Failed to download '{name}': {e}")
    except OSError as e:
        print(f"  -> Could not save file '{name}': {e}")


def download_file(download_link, save_path="."):
    """Downloads a file or all files in a folder from a GoFile link."""
    print(f"Attempting to download from: {download_link}")

    try:
        # Extract content ID from the URL
        try:
            content_id = download_link.split('/')[-1]
            if not content_id:
                raise IndexError
        except IndexError:
            print("Error: Invalid GoFile URL format. Could not extract content ID.")
            return

        api_url = f"https://api.gofile.io/contents/{content_id}"
        content_response = requests.get(api_url)
        content_response.raise_for_status()
        response_data = content_response.json()

        if response_data.get("status") != "ok":
            print("Error: GoFile API returned a non-ok status.")
            print("--- Server Response ---")
            print(content_response.text)
            print("-----------------------")
            return

        content_data = response_data.get("data")
        if not content_data:
            print("Error: 'data' field not found in API response.")
            print("--- Server Response ---")
            print(content_response.text)
            print("-----------------------")
            return

        content_type = content_data.get("type")
        if content_type == "folder":
            children = content_data.get("children", {})
            if not children:
                print("The folder is empty. Nothing to download.")
                return

            folder_name = content_data.get("name", content_id)
            print(f"\n--- Found Folder: '{folder_name}' ---")
            print(f"It contains {len(children)} item(s).")
            folder_save_path = os.path.join(save_path, folder_name)

            for file_details in children.values():
                if file_details.get("type") == "file" and file_details.get("link"):
                    download_single_file(file_details["link"], file_details["name"], folder_save_path)

            print("---------------------------------")

        elif content_type == "file":
            print("\n--- Found File ---")
            download_single_file(content_data["link"], content_data["name"], save_path)
            print("--------------------")

        else:
            print(f"Error: Unrecognized content type '{content_type}'.")
            print("--- Server Response ---")
            print(content_response.text)
            print("-----------------------")

    except requests.exceptions.RequestException as e:
        print(f"Error during download request: {e}")
    except requests.exceptions.JSONDecodeError:
        print("Error: Failed to decode server response as JSON.")
        if 'content_response' in locals():
            print("--- Server Response ---")
            print(content_response.text)
            print("-----------------------")
    except (KeyError, IndexError):
        print("Error: Could not find the file or parse the download information.")
        print("Please check the link and make sure the content hasn't been deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and download files using the GoFile API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a file.")
    upload_parser.add_argument("file_path", help="The path to the file you want to upload.")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a file or folder.")
    download_parser.add_argument("download_link", help="The GoFile download link (e.g., https://gofile.io/d/xxxxxx).")
    download_parser.add_argument("--save_path", default=".",
                                 help="The directory where you want to save the downloaded content (optional).")

    args = parser.parse_args()

    if args.command == "upload":
        upload_file(args.file_path)
    elif args.command == "download":
        download_file(args.download_link, args.save_path)