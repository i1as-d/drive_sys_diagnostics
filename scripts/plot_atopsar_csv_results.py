import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import time
import re

# Global variable for the file path
# FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/foxy_full_loop_no_cam_/foxy_full_loop_no_cam_resources.csv"
# GPU_FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/foxy_full_loop_no_cam_/foxy_full_loop_no_cam_gpu.txt"
# DRIVE_DELAY_FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/foxy_full_loop_no_cam_/foxy_sensor_2_frenet_delay.txt"
# old_smi = True

FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/humble_full_loop_no_cam_/humble_full_loop_no_cam_resources.csv"
GPU_FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/humble_full_loop_no_cam_/humble_full_loop_no_cam_gpu.txt"
DRIVE_DELAY_FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/humble_full_loop_no_cam_/humble_sensor_2_frenet_delay.txt"

# FILE_PATH = "/home/deniz-a/Desktop/drive_sys_diagnostics/data/humble_monitoring_full_loop/humble_full_loop_bf_resources.csv"


def read_atop_csv(file_path):
    """Fetch data from csv file to python object

    Args:
        file_path (string): path of the csv file

    Returns:
        list: full data from the csv table
    """
    data_blocks = []
    with open(file_path, 'r') as file:
        block_lines = []
        for line in file:
            line = line.strip()
            if line.startswith('--------------------------'):
                if block_lines:
                    data_blocks.append('\n'.join(block_lines))
                    block_lines = []
            else:
                block_lines.append(line)
        if block_lines:
            data_blocks.append('\n'.join(block_lines))
    return data_blocks

def parse_gpu_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        skip_header = True
        for line in file:
            if skip_header:
                skip_header = False
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 12:  # ensuring all columns are present
                date_str, time_str, gpu, pwr, gtemp, _, sm, mem, enc, dec, mclk, pclk = parts
                timestamp = time_str  # keeping only the time part
                row = [timestamp, int(gpu), float(pwr), float(gtemp), '-', float(sm), float(mem), float(enc), float(dec), float(mclk), float(pclk)]
                data.append(row)
            if len(parts) == 14:  # ensuring all columns are present
                date_str, time_str, gpu, pwr, gtemp, _, sm, mem, enc, dec, jpg, ofa, mclk, pclk = parts
                timestamp = time_str  # keeping only the time part
                row = [timestamp, int(gpu), float(pwr), float(gtemp), '-', float(sm), float(mem), float(enc), float(dec), float(jpg), float(ofa), float(mclk), float(pclk)]
                data.append(row)
    return np.array(data, dtype=object)

def plot_gpu_data(gpu_data, old_smi):
    # Extracting columns
    timestamps = gpu_data[:, 0]
    pwr = gpu_data[:, 2].astype(float)
    gtemp = gpu_data[:, 3].astype(float)
    sm = gpu_data[:, 5].astype(float)
    mem = gpu_data[:, 6].astype(float)
    enc = gpu_data[:, 7].astype(float)
    dec = gpu_data[:, 8].astype(float)
    if old_smi:
        jpg = np.array([])
        ofa = np.array([])
        mclk = gpu_data[:, 9].astype(float)
        pclk = gpu_data[:, 10].astype(float)
    else:
        jpg = gpu_data[:, 9].astype(float)
        ofa = gpu_data[:, 10].astype(float)
        mclk = gpu_data[:, 11].astype(float)
        pclk = gpu_data[:, 12].astype(float)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(timestamps, gtemp, label='gtemp (C)')
    ax1.plot(timestamps, pwr, label='pwr (W)')
    ax1.plot(timestamps, sm, label='sm (%)')
    ax1.plot(timestamps, mem, label='mem (%)')
    # ax1.plot(timestamps, enc, label='enc (%)')
    # ax1.plot(timestamps, dec, label='dec (%)')
    # ax1.plot(timestamps, jpg, label='jpg (%)')
    # ax1.plot(timestamps, ofa, label='ofa (%)')
    ax1.set_ylabel('Temp, Pow, loads (legend)')
    ax1.set_title('Temp, Power, GPU, memory')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(timestamps, mclk, label='mclk')
    ax2.plot(timestamps, pclk, label='pclk')
    ax2.set_ylabel('clock speed (Hz)')
    ax2.set_title('Memory and pixel clock speeds')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Formatting
    plt.xlabel('Timestamp')
    plt.xticks(timestamps[::len(timestamps)//30], rotation=45)  # Adjusting tick frequency

    # Displaying the plot
    plt.tight_layout()
    plt.show()

def process_thread_block(data):
    lines = data.strip().split('\n')
    column_titles = lines[0].split()[1:-1]
    matrix = []
    
    for line in lines[1:]:
        elements = (line.split())
        matrix.append(elements[0:])

    # Transpose the matrix
    transposed_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    return column_titles, transposed_matrix

def parse_simple_data_block(data):
    lines = data.strip().split('\n')
    column_titles = lines[0].split()[1:-1]
    timestamps = []
    matrix = []
    last_timestamp = ''
    
    for line in lines[1:]:
        elements = (line.split())
        if elements[0][:2].isdigit():
            timestamps.append(elements[0])
            matrix.append(elements[1:])
            last_timestamp = elements[0]
        else:
            matrix.append([last_timestamp] + elements)

    # Transpose the matrix
    transposed_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    return timestamps, column_titles, transposed_matrix

def remove_percentage(column):
    new_column = []
    for element in column:
        if isinstance(element, str) and '%' in element:
            element = element.rstrip('%')
        new_column.append(element)
    return new_column

def process_disk_block(data):
    lines = data.strip().split('\n')
    
    # Extract column titles and initialize matrix
    first_line = lines[0].split()
    column_titles = first_line[1:-1]
    last_timestamp = first_line[0]
    matrix = np.full((len(lines) - 1, 10), '', dtype=object)
    
    for i, line in enumerate(lines[1:]):
        elements = line.split()
        if len(elements) != 9:
            # Update each element individually
            for j, element in enumerate(elements):
                matrix[i, j] = element
            last_timestamp = elements[0]
        else:
            # Update each element individually
            matrix[i, 0] = last_timestamp
            for j in range(1, len(elements) + 1):
                matrix[i, j] = elements[j - 1]

    # further processing
    matrix[matrix == ''] = np.nan
    matrix = matrix[:, :-1]
    matrix[:, 2] = remove_percentage(matrix[:, 2])
    matrix[:, 2:] = matrix[:, 2:].astype(float)

    print(matrix)

    return column_titles, matrix

def parse_nested_data_block(data):
    lines = data.strip().split('\n')
    column_titles = lines[0].split()[1:-1]

    timestamps = []
    tensor = []
    current_matrix = []

    for line in lines[1:]:
        elements = (line.split())
        if len(elements) == 11:  # Check if it's a timestamp line
            # push back prev matrix in tensor
            tensor.append(current_matrix)
            # re-init matrix
            current_matrix = [] 
            # only take the rest
            timestamps.append(elements[0])
            current_matrix.append(elements[1:])
        else:
            current_matrix.append(elements)

    # remove first matrix pushed empty
    tensor = tensor[1:]
    timestamps = timestamps[1:]

    nb_matrices = len(tensor)
    col_sz = len(tensor[0][:][0])
    row_sz = len(tensor[:][0])
    print("Size of timestamp vector :", len(timestamps))
    print("Size of tensor:", nb_matrices, " matrices of size ", row_sz, col_sz)

    return timestamps, column_titles, tensor

def plot_cpu_data(timestamps, column_titles, tensor):
    timestamp_vector = range(len(timestamps))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Assuming num_cores is the number of CPU cores
    num_cores = len(tensor[:][0])-1  # Change this to the actual number of CPU cores

    first_interest_indices = [1, 3, 9]
    for idx in first_interest_indices:
        values = [int(matrix[0][idx]) if matrix[0][idx].isdigit() else matrix[0][idx] for matrix in tensor]
        values = np.array(values, dtype=float)
        values /= num_cores
        axes[0].plot(timestamp_vector, values, label=column_titles[idx], linestyle='-', alpha=0.8, marker='')

    axes[0].set_ylabel('time spent (%)')
    axes[0].set_title('Average time spent of cpu in user, idle and kernel spaces')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(timestamp_vector[::len(timestamp_vector)//30])
    axes[0].set_xticklabels(timestamps[::len(timestamps)//30], rotation='vertical')
    axes[0].legend()

    # Plot remaining values against timestamp in the second subplot
    remaining_indices = [2, 4, 5, 8]  # Indices of the remaining columns
    for idx, color in zip(remaining_indices, ['green', 'blue', 'orange', 'red']):
        values = [int(matrix[0][idx]) if matrix[0][idx].isdigit() else matrix[0][idx] for matrix in tensor]
        values = np.array(values, dtype=float)
        values /= num_cores
        axes[1].plot(timestamp_vector, values, label=column_titles[idx], color=color, linestyle='-', alpha=0.8, marker='')

    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('time spent (%)')
    axes[1].set_title('Average time spent modifying prio, HW and SW interrupts, or waiting')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(timestamp_vector[::len(timestamp_vector)//30])
    axes[1].set_xticklabels(timestamps[::len(timestamps)//30], rotation='vertical')
    axes[1].legend()

    # Adding a vertical line at the tenth timestamp on both graphs
    launch_timestamp_index = 55
    goal_pose_timestamp_index = 95
    axes[0].axvline(x=launch_timestamp_index, color='k', linestyle='--')
    axes[1].axvline(x=launch_timestamp_index, color='k', linestyle='--')
    axes[0].axvline(x=goal_pose_timestamp_index, color='k', linestyle='--')
    axes[1].axvline(x=goal_pose_timestamp_index, color='k', linestyle='--')
    

    plt.tight_layout()
    plt.show()

def plot_columns(timestamps, column_titles, matrix_of_columns, indices, shared_subplots, y_labels):
    num_subplots = len(shared_subplots) + len(set(indices) - set.union(*map(set, shared_subplots))) 
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots), sharex=True)
    if num_subplots == 1:
        axes = [axes]  # Ensure axes is a list if only one subplot
    
    subplot_idx = 0
    for subplot_indices, ylabel in zip(shared_subplots, y_labels):
        ax = axes[subplot_idx]
        for idx in subplot_indices:
            column_values = [int(re.findall(r'\d+', value)[0]) for value in matrix_of_columns[idx]]
            ax.plot(timestamps, column_values, label=column_titles[idx], linestyle='-', alpha=0.8, marker='.')
            ax.set_ylabel(ylabel)
            ax.set_title('')
            ax.legend()
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        subplot_idx += 1
    
    remaining_indices = set(indices) - set.union(*map(set, shared_subplots))
    for idx, ylabel in zip(remaining_indices, y_labels[len(shared_subplots):]):
        ax = axes[subplot_idx]
        column_values = [int(re.findall(r'\d+', value)[0]) for value in matrix_of_columns[idx]]
        ax.plot(timestamps, column_values, label=column_titles[idx], linestyle='-', alpha=0.8, marker='.')
        ax.set_ylabel(ylabel)
        ax.set_title('')
        ax.legend()
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        subplot_idx += 1
    
    # Limit the number of x-axis ticks to 10 or less
    num_ticks = min(10, len(timestamps))
    for ax in axes:
        ax.set_xticks(timestamps[::len(timestamps)//num_ticks])
    
    plt.xlabel('Timestamps')
    plt.tight_layout()
    plt.show()

def plot_disk_data(matrix, column_titles):
    # Extract relevant data
    timestamps = matrix[:, 0]
    disk_names = matrix[:, 1]
    data = matrix[:, 2:].astype(float)
    
    # Convert disk_names to string to handle NaNs
    disk_names = np.array(disk_names, dtype=str)
    
    unique_disks = np.unique(disk_names[disk_names != 'nan'])
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Disk Data')
    
    # Set light grid for all subplots
    for ax in axs:
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5, color='gray')
        ax.set_xticks(np.linspace(0, len(timestamps) - 1, 30, dtype=int))
        ax.set_xticklabels(timestamps[np.linspace(0, len(timestamps) - 1, 30, dtype=int)], rotation=90)
    
    for disk in unique_disks:
        # Find indices corresponding to the disk
        disk_indices = np.where(disk_names == disk)[0]
        
        # Extract data for this disk
        disk_data = data[disk_indices]
        disk_timestamps = timestamps[disk_indices]
        
        # Subplot 1: busy against timestamp
        axs[0].plot(disk_timestamps, disk_data[:, 0], linestyle='--', marker='.', label=f'{disk} busy')
        
        # Subplot 2: KB/read and KB/writ against timestamp
        axs[1].plot(disk_timestamps, disk_data[:, 1], linestyle='--', marker='.', label=f'{disk} KB/read')
        axs[1].plot(disk_timestamps, disk_data[:, 2], linestyle='--', marker='.', label=f'{disk} KB/writ')
        
        # Subplot 3: avque and avserv against timestamp
        axs[2].plot(disk_timestamps, disk_data[:, 3], linestyle='--', marker='.', label=f'{disk} avque')
        axs[2].plot(disk_timestamps, disk_data[:, 4], linestyle='--', marker='.', label=f'{disk} avserv')
    
    # Set y-axis labels
    axs[0].set_ylabel('Busy (%)')
    axs[1].set_ylabel('Data per r/w (KB)')
    axs[2].set_ylabel('avg service')
    
    # Set common x-axis label
    axs[-1].set_xlabel('Timestamp')
    
    # Add second y-axis for the last subplot
    axs[2].twinx().set_ylabel('nb waiting I/O reqs (-)')
    axs[2].set_ylabel('Service time (ms)')
    
    # Show legends
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    # Show plot
    plt.show()

def plot_delay_data(file_path):
    # Read the file and extract floating-point numbers
    with open(file_path, 'r') as file:
        delay_data = [float(line.split(': ')[1]) for line in file.readlines()]

    # Create a vector of the same size as delay_data
    generic_vector = np.arange(len(delay_data))

    # Create a figure with two subplots: one for the delay data and one for the histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot delay_data against the generic vector
    ax1.plot(generic_vector, delay_data, label='Delay Data', linestyle='-', alpha=0.8, marker='.')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('capture-to-frenet delay (s)')
    ax1.set_title('Pipeline delay during the drive')
    ax1.legend()

    # Plot histogram of encountered y values
    n, bins, patches = ax2.hist(delay_data, bins=20, edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Delay Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Delay Values')

    # Calculate mean and variance
    mean_delay = np.mean(delay_data)
    variance_delay = np.var(delay_data)

    # Add vertical lines for mean and variance
    ax2.axvline(x=mean_delay, color='red', linestyle='--', label=f'Mean: {mean_delay:.2f}')
    ax2.axvline(x=mean_delay - np.sqrt(variance_delay), color='blue', linestyle='--', label=f'Std Dev: {np.sqrt(variance_delay):.2f}')
    ax2.axvline(x=mean_delay + np.sqrt(variance_delay), color='blue', linestyle='--')

    # Add legend
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

    # Return delay_data vector
    return delay_data

if __name__ == "__main__":

    # ========== delay report ==========
    plot_delay_data(DRIVE_DELAY_FILE_PATH)

    # ========== gpu report ==========
    gpu_data = parse_gpu_file(GPU_FILE_PATH)
    plot_gpu_data(gpu_data, old_smi)


    # ========== atop report ========== 
    # get all data blocks
    data_blocks = read_atop_csv(FILE_PATH)
    print("Blocks analyzed (excl. header block) : ", len(data_blocks) - 1)

    # data_blocks[:][0] # HEADER
    # data_blocks[:][1] # cpu perfs
    # data_blocks[:][2] # avrg workload
    # data_blocks[:][3] # thread related metrics
    # data_blocks[:][4] # memory
    # data_blocks[:][5] # disk I/O activity
    # data_blocks[:][6] # network activity

    # ----- 1 : CPU perfs
    timestamps, column_titles, tensor = parse_nested_data_block(data_blocks[1])
    print("timestamps vector of strings is of size", len(timestamps))
    print("column titles : ", column_titles)
    plot_cpu_data(timestamps, column_titles, tensor)

    # ----- 2 : avg workload
    column_titles, matrix = process_thread_block(data_blocks[2])
    plot_columns(matrix[0], column_titles, matrix[1:], [0,1,2,3,4,5], [[0,1,2],[3,4,5]],["nb requests per sec (-)","total queued processes"])

    # # ----- 4 : memory
    timestamps, column_titles, matrix_of_columns = parse_simple_data_block(data_blocks[4])
    plot_columns(timestamps, column_titles, matrix_of_columns, [2,3,4,5], [[4,5]], ["Memory (Mb)","Memory (Mb)","Memory (Mb)"])

    # # ----- 5 : disk I/O
    column_titles, matrix = process_disk_block(data_blocks[5])
    plot_disk_data(matrix, column_titles)

    # # ----- 6 : network traffic
    timestamps, column_titles, matrix_of_columns = parse_simple_data_block(data_blocks[6])
    plot_columns(timestamps, column_titles, matrix_of_columns, [0,1,2,3,4,5], [[0,1,2],[3,4,5]], ["Packets transfer rate (Hz)", "Packets transfer rate (Hz)"])
