import subprocess
import re
import time
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description="Monitor and plot NVIDIA GPU stats to a PNG file.")
parser.add_argument("--gpu-util", action="store_true", default=True, help="Monitor GPU utilization")
parser.add_argument("--mem-util", action="store_true", default=True, help="Monitor memory utilization")
parser.add_argument("--temp", action="store_true", default=True, help="Monitor temperature")
parser.add_argument("--filename", type=str, default="output.png", help="Filename for the output PNG")
args = parser.parse_args()

# Regex patterns to extract data
patterns = {
    "gpu_util": re.compile(r"(\d+)%\s+Default\s+"),
    "mem_util": re.compile(r"(\d+)%\s+\|\s+\d+MiB\s+/\s+\d+MiB\s+\|"),
    "temp": re.compile(r"(\d+)C\s+\|\s+")
}

# Data storage
data = {
    "time": [],
    "gpu_util": [],
    "mem_util": [],
    "temp": [],
}

# Function to parse output
def parse_output(output):
    if args.gpu_util:
        gpu_util_match = patterns["gpu_util"].search(output)
        if gpu_util_match:
            data["gpu_util"].append(int(gpu_util_match.group(1)))
        else:
            data["gpu_util"].append(None)
    
    if args.mem_util:
        mem_util_match = patterns["mem_util"].search(output)
        if mem_util_match:
            data["mem_util"].append(int(mem_util_match.group(1)))
        else:
            data["mem_util"].append(None)
    
    if args.temp:
        temp_match = patterns["temp"].search(output)
        if temp_match:
            data["temp"].append(int(temp_match.group(1)))
        else:
            data["temp"].append(None)

# Main loop to continuously capture output
try:
    start_time = time.time()
    process = subprocess.Popen(["nvidia-smi", "-l", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            current_time = time.time() - start_time
            data["time"].append(current_time)
            parse_output(output.strip())

except KeyboardInterrupt:
    print("Stopping and preparing data for plotting...")

finally:
    process.terminate()
    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    
    # Plotting
    sns.set()
    plt.figure(figsize=(10, 6))
    
    if args.gpu_util:
        sns.lineplot(data=df, x="time", y="gpu_util", label="GPU Utilization")
    if args.mem_util:
        sns.lineplot(data=df, x="time", y="mem_util", label="Memory Utilization")
    if args.temp:
        sns.lineplot(data=df, x="time", y="temp", label="Temperature")
    
    plt.title("NVIDIA GPU Metrics Over Time")
    plt.ylabel("Value")
    plt.xlabel("Time (s)")
    plt.legend()
    
    # Save the plot to a PNG file
    plt.savefig(args.filename)
    print(f"Plot saved to {args.filename}")
