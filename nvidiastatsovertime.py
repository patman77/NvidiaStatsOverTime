import subprocess
import re
import time
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description="Monitor and plot NVIDIA GPU stats to a PNG file with a timeout.")
parser.add_argument("--gpu-util", action="store_true", default=True, help="Monitor GPU utilization")
parser.add_argument("--mem-util", action="store_true", default=True, help="Monitor memory utilization")
parser.add_argument("--temp", action="store_true", default=True, help="Monitor temperature")
parser.add_argument("--filename", type=str, default="output.png", help="Filename for the output PNG")
parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds for the monitoring")
args = parser.parse_args()

# Regex patterns to extract data
patterns = {
    "gpu_util": re.compile(r"\|\s+(\d+)%\s+Default\s+\|"),
    "mem_util": re.compile(r"\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|"),
    "temp": re.compile(r"\|\s+(\d+)C\s+P\d+\s+\|")  # Make sure this regex matches the output
}

# Data collection function
def parse_output(output):
    new_row = {}
    if args.gpu_util:
        gpu_util_match = patterns["gpu_util"].search(output)
        new_row["gpu_util"] = int(gpu_util_match.group(1)) if gpu_util_match else None
    
    if args.mem_util:
        mem_util_match = patterns["mem_util"].search(output)
        if mem_util_match:
            used_memory, total_memory = mem_util_match.groups()
            new_row["mem_util"] = (int(used_memory) / int(total_memory)) * 100
        else:
            new_row["mem_util"] = None
    
    if args.temp:
        temp_match = patterns["temp"].search(output)
        new_row["temp"] = int(temp_match.group(1)) if temp_match else None
        print(f"Debug - Parsed temperature: {new_row['temp']}C")  # Debug print
    
    return new_row

def monitor_and_collect_data(timeout):
    start_time = time.time()
    collected_data = []
    process = subprocess.Popen(["nvidia-smi", "-l", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        while time.time() - start_time < timeout:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and "MiB /" in output:  # Ensures we're parsing a line with the needed info
                new_row = parse_output(output.strip())
                new_row["time"] = time.time() - start_time  # Add time to each row
                collected_data.append(new_row)
    finally:
        process.terminate()
    
    return pd.DataFrame(collected_data)

# Main execution
df = monitor_and_collect_data(args.timeout)

# Debug print to inspect the DataFrame before plotting
print(df)

# Plotting
if not df.empty:
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))

    if 'gpu_util' in df.columns:
        plt.plot(df['time'], df['gpu_util'], label='GPU Utilization')
    if 'mem_util' in df.columns:
        plt.plot(df['time'], df['mem_util'], label='Memory Utilization')
    if 'temp' in df.columns:
        plt.plot(df['time'], df['temp'], label='Temperature')  # Plot temperature

    plt.title('NVIDIA GPU Metrics Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(args.filename)
    print(f"Plot saved to {args.filename}")
else:
    print("No data collected.")
