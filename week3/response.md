Here's a Python script that generates synthetic hydrophone sensor data according to your specifications. It uses common libraries like NumPy, Pandas, and Matplotlib to create and visualize the data. 

Make sure to have the necessary libraries installed before running the script. You can install them using pip if you haven't done so already:

```bash
pip install numpy pandas matplotlib
```

Here’s the script:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a directory for datasets
os.makedirs('datasets', exist_ok=True)

# Constants
sampling_rate = 250e6  # 250 MHz
sample_time = 50e-6    # 50µs
num_samples = int(sampling_rate * sample_time)  # Total number of samples
time = np.linspace(0, sample_time, num_samples)  # Time vector

# Function to generate synthetic data
def generate_data():
    # Generating random durations and values
    blast_rise_time = np.random.normal(loc=15e-9, scale=2e-9)  # Fast rise time for blast wave
    blast_decay_time = np.random.normal(loc=1.5e-6, scale=0.2e-6)  # Slower decay
    blast_rarefaction_time = np.random.normal(loc=7.5e-6, scale=1e-6)  # Longer rarefaction

    voltage_rise_time = np.random.uniform(100e-9, 200e-9)  # Fast rise for voltage pulse
    voltage_fall_time = np.random.uniform(1e-6, 3e-6)  # Slow fall
    
    current_rise_time = np.random.uniform(400e-9, 600e-9)  # Fast rise for current pulse
    current_fall_time = np.random.uniform(750e-9, 1500e-9)  # Slow fall

    # Generate Channel A: Blast wave (peak: 0.5 to 1.5V)
    blast_peak = np.random.uniform(0.5, 1.5)
    blast_wave = np.zeros(num_samples)

    # Define timing for channels
    blast_start_time = int((10e-6 + np.random.uniform(1e-6, 3e-6)) * sampling_rate)
    blast_peak_time = blast_start_time + int(blast_rise_time * sampling_rate)
    blast_end_time = blast_peak_time + int(blast_decay_time * sampling_rate)
    rarefaction_end_time = blast_end_time + int(blast_rarefaction_time * sampling_rate)

    blast_wave[blast_start_time:blast_peak_time] = np.linspace(0, blast_peak, blast_peak_time - blast_start_time)
    blast_wave[blast_peak_time:blast_end_time] = np.linspace(blast_peak, -0.1, blast_end_time - blast_peak_time)
    blast_wave[blast_end_time:rarefaction_end_time] = np.linspace(-0.1, -0.4, rarefaction_end_time - blast_end_time)

    # Generate Channel B: Voltage pulse (3000V to 4000V)
    voltage_peak = np.random.uniform(3000, 4000)
    voltage_pulse = np.zeros(num_samples)

    voltage_start_time = int(10e-6 * sampling_rate)
    voltage_end_time = voltage_start_time + int(5e-6 * sampling_rate)

    voltage_pulse[voltage_start_time:voltage_start_time + int(voltage_rise_time * sampling_rate)] = np.linspace(0, voltage_peak, int(voltage_rise_time * sampling_rate))
    voltage_pulse[voltage_start_time + int(voltage_rise_time * sampling_rate):voltage_end_time] = np.linspace(voltage_peak, 0, voltage_end_time - (voltage_start_time + int(voltage_rise_time * sampling_rate)))

    # Generate Channel C: Current pulse (130A to 180A)
    current_peak = np.random.uniform(130, 180)
    current_pulse = np.zeros(num_samples)

    current_start_time = voltage_end_time
    current_end_time = current_start_time + int(5e-6 * sampling_rate)

    current_pulse[current_start_time:current_start_time + int(current_rise_time * sampling_rate)] = np.linspace(0, current_peak, int(current_rise_time * sampling_rate))
    current_pulse[current_start_time + int(current_rise_time * sampling_rate):current_end_time] = np.linspace(current_peak, 0, current_end_time - (current_start_time + int(current_rise_time * sampling_rate)))

    # Scale the outputs of channels to be in the range of -1 to 5
    blast_wave_scaled = (blast_wave - blast_wave.min()) / (blast_wave.max() - blast_wave.min()) * 6 - 1
    voltage_pulse_scaled = (voltage_pulse - voltage_pulse.min()) / (voltage_pulse.max() - voltage_pulse.min()) * 6 - 1
    current_pulse_scaled = (current_pulse - current_pulse.min()) / (current_pulse.max() - current_pulse.min()) * 6 - 1

    return time, blast_wave_scaled, voltage_pulse_scaled, current_pulse_scaled

# Generate and save datasets
for i in range(100):
    time, channel_a, channel_b, channel_c = generate_data()
    data = pd.DataFrame({
        'time': time,
        'Channel A': channel_a,
        'Channel B': channel_b,
        'Channel C': channel_c
    })
    data.to_csv(f'datasets/dataset_{i+1}.csv', index=False)

# Load the last dataset for plotting
last_dataset = pd.read_csv('datasets/dataset_100.csv')

# Plotting the channels
plt.figure(figsize=(15, 7))
plt.plot(last_dataset['time'], last_dataset['Channel A'], label='Channel A (Blast Wave)', alpha=0.7)
plt.plot(last_dataset['time'], last_dataset['Channel B'], label='Channel B (Voltage Pulse)', alpha=0.7)
plt.plot(last_dataset['time'], last_dataset['Channel C'], label='Channel C (Current Pulse)', alpha=0.7)
plt.title('Synthetic Hydrophone Sensor Data')
plt.xlabel('Time (s)')
plt.ylabel('Scaled Voltage / Current')
plt.legend()
plt.grid()
plt.savefig('datasets/last_dataset_plot.png')  # Save the figure for review
plt.show()
```

### Explanation of the Code:
1. **Directory Creation**: Creates a folder named 'datasets' for storing generated CSV files.
2. **Constants**: Defines constants for sampling rate and sample time to calculate the number of samples.
3. **Data Generation**: The `generate_data` function generates synthetic sensor data based on the specified parameters, utilizing random variations.
4. **Saving Datasets**: A loop generates 100 datasets, saving each to a CSV file.
5. **Plotting**: The last dataset is loaded and plotted using Matplotlib. The channels are scaled and visible with a legend.

### Final Notes:
- Adjust the mean and standard deviations of the normally distributed functions according to your requirements.
- The resulting datasets are saved as CSV files, and the last dataset's plot is displayed using Matplotlib. You may further refine the visual aesthetics as needed.