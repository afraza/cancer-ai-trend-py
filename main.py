import importlib
import os


def list_preprocessing_functions():
    preprocessing_dir = 'preprocessing'
    preprocessing_files = [f for f in os.listdir(preprocessing_dir) if f.endswith('.py') and f != '__init__.py']
    return [os.path.splitext(f)[0] for f in preprocessing_files]


def run_preprocessing_function(function_name):
    try:
        module = importlib.import_module(f'preprocessing.{function_name}')
        if hasattr(module, 'run'):
            module.run()
        else:
            print(f"Error: {function_name} does not have a 'run' function.")
    except ImportError:
        print(f"Error: Could not import {function_name}")
    except Exception as e:
        print(f"Error running {function_name}: {str(e)}")


def main():
    preprocessing_functions = list_preprocessing_functions()

    print("Available preprocessing functions:")
    for i, func in enumerate(preprocessing_functions, 1):
        print(f"{i}. {func}")

    while True:
        choice = input("\nEnter the number of the function you want to run (or 'q' to quit): ")
        if choice.lower() == 'q':
            break

        try:
            choice = int(choice)
            if 1 <= choice <= len(preprocessing_functions):
                selected_function = preprocessing_functions[choice - 1]
                print(f"\nRunning {selected_function}...")
                run_preprocessing_function(selected_function)
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


if __name__ == "__main__":
    main()
