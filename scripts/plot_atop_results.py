import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to atop log file
folder_name = "../data/humble_full_loop_no_cam_"
atop_file_name = 'humble_full_loop_no_cam_resources.csv'

# Path to Nvidia GPU usage file
nv_gpu_log_file_name = 'humble_full_loop_no_cam_gpu.txt'

# Function to parse atop log file
# Function to parse atop log file
def parse_atop_log(file_path):
    cpu_usage = []
    memory_usage = []
    network_in = []
    network_out = []
    times = []

    try:
        with open(file_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('---') or line.startswith('deniz-a'):
                    continue
                if 'analysis date' in line:
                    continue
                if ':' not in line:  # Skip lines without time information
                    continue
                parts = line.split()  # Split the line into parts
                time_str = parts[0]  # Extract time string
                cpu_str = parts[1]  # Extract CPU usage string
                memory_str = parts[2]  # Extract memory usage string
                net_in_str = parts[3]  # Extract network in string
                net_out_str = parts[4]  # Extract network out string

                # Convert to float, handling empty strings gracefully
                try:
                    cpu_usage.append(float(cpu_str))
                    memory_usage.append(float(memory_str))
                    network_in.append(float(net_in_str))
                    network_out.append(float(net_out_str))
                except ValueError:
                    # Skip this line if conversion fails
                    continue

                times.append(time_str)
    except Exception as e:
        print("An exception occurred:", e)
        return [], [], [], [], []

    return times, cpu_usage, memory_usage, network_in, network_out

# Function to parse Nvidia GPU usage file
def parse_nvidia_gpu_usage(file_path):
    times = []
    gpu_usage = []

    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                line = line.strip()
                if line:  # Check if the line is not empty
                    if line[0].isdigit():  # Check if the line starts with a digit (indicating it contains data)
                        parts = line.split()
                        date_str = parts[0]  # Extract date string
                        time_str = parts[1]  # Extract time string
                        gpu_usage.append(float(parts[2]))
                        times.append(time_str)
    except Exception as e:
        print("An exception occurred. Have you recorded nvidia gpu activity ?", e)

    return times, gpu_usage


# Parse atop log file
times, cpu_usage, memory_usage, network_in, network_out = parse_atop_log(folder_name + "/" + atop_file_name)

print("Parsed data:")
print("Times:", times)
print("CPU Usage:", cpu_usage)
print("Memory Usage:", memory_usage)
print("Network In:", network_in)
print("Network Out:", network_out)

# Parse Nvidia GPU usage file
gpu_times, gpu_usage = parse_nvidia_gpu_usage(folder_name + "/" + nv_gpu_log_file_name)

# Convert time strings to indices
time_indices = range(len(times))
gpu_time_indices = range(len(gpu_times))

# Find the index for the middle point
middle_index = len(times) // 2

# Function to compute the integral over time
def compute_integral_over_time(times, values):
    # Convert times to seconds
    time_seconds = [int(t.split(':')[0]) * 3600 + int(t.split(':')[1]) * 60 + int(t.split(':')[2]) for t in times]
    
    # Compute the integral using trapezoidal rule
    integral = np.trapz(values, x=time_seconds)
    
    return integral

# Function to compute the average usage
def compute_average_usage(times, values):
    total_duration = (int(times[-1].split(':')[0]) * 3600 + int(times[-1].split(':')[1]) * 60 + int(times[-1].split(':')[2])) - \
                     (int(times[0].split(':')[0]) * 3600 + int(times[0].split(':')[1]) * 60 + int(times[0].split(':')[2]))
    
    integral = compute_integral_over_time(times, values)
    average = integral / total_duration
    
    return average

# Compute the average CPU usage
cpu_average = compute_average_usage(times, cpu_usage)
# Compute the average memory usage
memory_average = compute_average_usage(times, memory_usage)
# Compute the average network usage
network_in_average = compute_average_usage(times, network_in)
network_out_average = compute_average_usage(times, network_out)
# Compute the average Nvidia GPU usage
gpu_average = compute_average_usage(gpu_times, gpu_usage)

# Plot CPU usage
plt.subplot(2, 2, 1)
plt.plot(time_indices, cpu_usage, marker='.', label=f'CPU Usage\n(Average: {cpu_average:.2f}%)', linestyle='-', alpha=0.8)
plt.title('CPU Usage')
plt.ylabel('Usage')
plt.xticks([time_indices[0], middle_index, time_indices[-1]], [times[0], times[middle_index], times[-1]])
plt.legend()
plt.grid(alpha=0.2)  # Add a light grid

# Plot memory usage
plt.subplot(2, 2, 2)
plt.plot(time_indices, memory_usage, marker='.', label=f'Memory Usage\n(Average: {memory_average:.2f}%)', linestyle='-', alpha=0.8)
plt.title('Memory Usage')
plt.ylabel('Usage')
plt.xticks([time_indices[0], middle_index, time_indices[-1]], [times[0], times[middle_index], times[-1]])
plt.legend()
plt.grid(alpha=0.2)  # Add a light grid

# Plot network usage
plt.subplot(2, 2, 4)
plt.plot(time_indices, network_in, label='Network In', marker='.', linestyle='-', alpha=0.8)
plt.plot(time_indices, network_out, label='Network Out', marker='.', linestyle='-', alpha=0.8)
plt.title('Network Usage')
plt.ylabel('Usage')
plt.xticks([time_indices[0], middle_index, time_indices[-1]], [times[0], times[middle_index], times[-1]])
plt.legend()
# Display average network usage in the legend
plt.legend([f'Network In\n(Average: {network_in_average:.2f}bytes/s)', f'Network Out\n(Average: {network_out_average:.2f}bytes/s)'])
plt.grid(alpha=0.2)  # Add a light grid

# Plot Nvidia GPU usage
plt.subplot(2, 2, 3)
plt.plot(gpu_time_indices, gpu_usage, label='GPU Usage', marker='.', linestyle='-', alpha=0.8)
plt.title('Nvidia GPU Usage')
plt.ylabel('Usage')
plt.xticks([gpu_time_indices[0], len(gpu_times) // 2, gpu_time_indices[-1]], [gpu_times[0], gpu_times[len(gpu_times) // 2], gpu_times[-1]])
plt.legend()
# Display average GPU usage in the legend
plt.legend([f'GPU Usage\n(Average: {gpu_average:.2f}%)'])
plt.grid(alpha=0.2)  # Add a light grid

plt.tight_layout()
plt.show()

