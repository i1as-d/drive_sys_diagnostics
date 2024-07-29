import os
import shutil
import subprocess
import sys

def main(input_data_path):
    # Validate input
    if not os.path.isfile(input_data_path):
        print(f"Error: The input path '{input_data_path}' is not a valid file.")
        sys.exit(1)

    # Define destination folder and file names
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination_folder = os.path.join(script_dir, 'drive_sys_diagnostics', 'data')
    os.makedirs(destination_folder, exist_ok=True)

    # Extract input file name and create new file name
    input_file_name = os.path.basename(input_data_path)
    new_file_name = os.path.splitext(input_file_name)[0] + '.atop'
    destination_file_path = os.path.join(destination_folder, new_file_name)

    # Step 1: Copy the .atop file
    try:
        shutil.copy(input_data_path, destination_file_path)
        print(f"Copied '{input_data_path}' to '{destination_file_path}'")
    except Exception as e:
        print(f"Error copying file: {e}")
        sys.exit(1)

    # Step 2: Run the atopsar command
    csv_file_name = os.path.splitext(new_file_name)[0] + '.csv'
    csv_file_path = os.path.join(destination_folder, csv_file_name)
    atopsar_command = [
        'atopsar', '-r', destination_file_path, '-w', '-m', '-d', '-p', '-c', '-P'
    ]
    
    try:
        with open(csv_file_path, 'w') as csv_file:
            subprocess.run(atopsar_command, stdout=csv_file, check=True)
        print(f"Generated CSV file at '{csv_file_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error running atopsar command: {e}")
        sys.exit(1)

    # Step 3: Run the plot script
    plot_script_path = os.path.join(script_dir, 'plot_atopsar_csv_results.py')
    plot_command = ['python3', plot_script_path, csv_file_path]
    
    try:
        subprocess.run(plot_command, check=True)
        print(f"Executed plot script '{plot_script_path}' with argument '{csv_file_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error running plot script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python automate_processing.py <input_data_path>")
        sys.exit(1)

    input_data_path = sys.argv[1]
    main(input_data_path)
