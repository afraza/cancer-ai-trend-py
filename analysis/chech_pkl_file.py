import pickle
import os


def load_and_show_records():
    # Ask the user for the file name
    file_name = input("Enter the name of the PKL file (including .pkl extension): ")

    # Construct the full path assuming the PKL file is in the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_dir, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        print("File not found! Please check the filename and try again.")
        return

    # Load and display records
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            print("\nFew records from the PKL file:")
            if isinstance(data, list) or isinstance(data, tuple):
                for record in data[:5]:  # Show first 5 records if it's a list/tuple
                    print(record)
            elif isinstance(data, dict):
                for i, (key, value) in enumerate(data.items()):
                    print(f"{key}: {value}")
                    if i == 4:  # Show first 5 items in a dictionary
                        break
            else:
                print("Unsupported data format. Unable to display records.")
    except Exception as e:
        print(f"Error loading file: {e}")


if __name__ == "__main__":
    load_and_show_records()
