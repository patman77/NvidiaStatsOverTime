
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
    "mem_util": re.compile(r"\|\s+(\d+)MiB / (\d+)MiB\s+\|"),
    "temp": re.compile(r"\|\s+(\d+)C\s+P\d+\s+\|")
}

def parse_output(output):
    matches = re.finditer(patterns["gpu_util"], output)
    data = []
    for match in matches:
        gpu_id = match.group(1)
        gpu_util = match.group(2)
        mem_util_match = patterns["mem_util"].search(output)
        temp_match = patterns["temp"].search(output)
        if mem_util_match and temp_match:
            used_memory, total_memory = mem_util_match.groups()
            temp = temp_match.group(1)
            data.append({
                "gpu_id": gpu_id,
                "gpu_util": int(gpu_util),
                "mem_util": (int(used_memory) / int(total_memory)) * 100,
                "temp": int(temp)
            })
    return data

def monitor_and_collect_data(timeout):
    start_time = time.time()
    collected_data = []
    process = subprocess.Popen(["nvidia-smi", "-l", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)

    try:
        while time.time() - start_time < timeout:
            output = process.stdout.read()
            if output == '' and process.poll() is not None:
                break
            data = parse_output(output)
            for d in data:
                d["time"] = time.time() - start_time
                collected_data.append(d)
    finally:
        process.terminate()

    return pd.DataFrame(collected_data)

# Main execution
df = monitor_and_collect_data(args.timeout)

# Plotting
if not df.empty:
    sns.set()
    plt.figure(figsize=(10, 6))
    for gpu_id in df['gpu_id'].unique():
        gpu_df = df[df['gpu_id'] == gpu_id]
        if 'gpu_util' in gpu_df.columns:
            sns.lineplot(data=gpu_df, x='time', y='gpu_util', label=f'GPU {gpu_id} Utilization')
        if 'mem_util' in gpu_df.columns:
            sns.lineplot(data=gpu_df, x='time', y='mem_util', label=f'GPU {gpu_id} Memory Utilization')
        if 'temp' in gpu_df.columns:
            sns.lineplot(data=gpu_df, x='time', y='temp', label=f'GPU {gpu_id} Temperature')

    plt.title('NVIDIA GPU Metrics Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(args.filename)
    print(f"Plot saved to {args.filename}")
else:
    print("No data collected.")
