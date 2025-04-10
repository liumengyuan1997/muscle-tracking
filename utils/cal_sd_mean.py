import numpy as np

# Read the file
values = []
with open('dice_scores_ibp.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')  # split by tab
        if len(parts) == 2:
            try:
                value = float(parts[1])
                values.append(value)
            except ValueError:
                pass  # skip lines with non-numeric values

# Convert to numpy array
values = np.array(values)

# Calculate mean and standard deviation
mean_val = np.mean(values)
std_val = np.std(values)

print(f"Mean: {mean_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")
