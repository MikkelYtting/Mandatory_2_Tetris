import numpy as np
from scipy.stats import ttest_rel, t

# Data from trained model
trained_blocks = [
    364, 886, 5105, 709, 1181, 3697, 3742, 2237, 3960, 1214, 808, 256, 1551, 1108, 823, 1651, 
    327, 693, 306, 512, 360, 370, 486, 1065, 5274, 648, 1226, 1009, 2199, 2172, 524, 619, 351, 181
]

dummy_blocks = [
    19, 22, 17, 18, 15, 14, 20, 22, 14, 20, 25, 12, 21, 17, 17, 23, 19, 18, 16, 15, 
    14, 20, 19, 15, 20, 15, 14, 22, 16, 18, 22, 20, 24, 21, 14, 20, 18, 17, 22, 19, 
    17, 18, 25, 14, 19, 22, 16, 22, 18, 16, 24, 15, 17, 21, 17, 16, 20, 15, 20, 18, 
    10, 17, 15, 17, 13, 24, 17, 17, 23, 22, 16, 20, 15, 13, 26, 15, 19, 17, 18, 18, 
    23, 20, 20, 17, 22, 21, 20, 22, 16, 15, 22, 18, 16, 20, 19, 16, 18, 16, 20, 20, 
    20, 18, 18, 17, 15, 18, 19, 18, 23, 18, 16, 17, 16, 16, 20, 19, 20, 20, 16, 22, 
    19, 18, 16, 19, 20, 13, 16, 17, 18, 17, 13, 18, 14, 21, 22, 18, 15, 18, 16, 16, 
    15, 18, 21, 21, 15, 21, 15, 20, 18, 11, 20, 14, 17, 21, 15, 15, 22, 17, 19, 23, 
    16, 23, 17, 19, 16, 16, 24, 17, 13, 17, 18, 17, 20, 16, 16, 18, 21, 21, 21, 14, 
    24, 20, 14, 21, 14, 21, 19, 16, 22, 29, 18, 21, 14, 14, 20, 20, 18, 20, 18, 18, 
    20, 19, 19, 23, 17, 15, 17, 19, 23, 19, 18, 17, 14, 19, 12, 26, 17, 20, 18, 22, 
    17, 20, 13, 12, 19, 21, 18, 18, 17, 22, 22, 16, 12, 20, 17, 17, 19, 20, 25, 19, 
    20, 20, 16, 15, 22, 20, 20, 18, 24, 16, 18, 18, 17, 18, 23, 18, 20, 24, 20, 19, 
    21, 21, 19, 15, 24, 18, 21, 24, 19, 18, 16, 18, 16, 15, 15, 17, 16, 10, 17, 22, 
    14, 16, 14, 15, 25, 16, 13, 15, 21, 15, 15, 19, 23, 19, 21, 19, 19, 19, 18, 17, 
    18, 16, 14, 15, 16, 17, 22, 16, 21, 15, 19, 19, 20, 19, 17, 19, 19, 16, 15, 19, 
    16, 15, 22, 18, 15, 18, 14, 20, 16, 21, 15, 12, 14, 18, 19, 17, 14, 17, 16, 17, 
    14, 18, 17, 17, 17, 19, 21, 26, 17, 15, 19, 17, 17, 19, 13, 23, 20, 10, 20, 21, 
    17, 16, 14, 17, 15, 22, 20, 18, 20, 25, 25, 19, 20, 24, 10, 22, 15, 16, 18, 22, 
    16, 24, 20, 24, 12, 18, 14, 24, 25, 19, 21, 21, 15, 17, 18, 17, 23, 17, 17, 21, 
    18, 18, 17, 19, 17, 20, 16, 24, 14, 24, 17, 21, 20, 17, 17, 20, 22, 14, 17, 22, 
    19, 17, 18, 14, 16, 18, 15, 16, 18, 20, 17, 15, 22, 20, 19, 17, 17, 25, 17, 16, 
    16, 20, 23, 17, 18, 11, 22, 17, 20, 19, 22, 19, 22, 24, 17, 19, 21, 23, 17, 15, 
    15, 17, 19, 17, 19, 20, 17, 17, 18, 22, 26, 23, 24, 15, 22, 20, 20, 15, 19, 21, 
    17, 21, 16, 23, 18, 17, 12, 14, 18, 20, 22, 17, 16, 14, 23, 22, 16, 15, 21, 16, 
    16, 22, 24, 15, 17, 15, 19, 19, 13, 23, 20, 19, 18, 16, 18, 23, 12, 21, 19, 19, 
    14, 21, 16, 24, 16, 18, 17, 28, 18, 20, 13, 19, 17, 16, 22, 20, 19, 19, 19, 20, 
    15, 18, 21, 22, 24
]

t_statistic, p_value = ttest_rel(trained_blocks, dummy_blocks[:len(trained_blocks)])

# Calculate the mean difference
mean_difference = np.mean(np.array(trained_blocks) - np.array(dummy_blocks[:len(trained_blocks)]))

# Calculate the standard error of the difference
std_error = np.std(np.array(trained_blocks) - np.array(dummy_blocks[:len(trained_blocks)]), ddof=1) / np.sqrt(len(trained_blocks))

# Calculate the confidence intervals
confidence_level = 0.95
degrees_freedom = len(trained_blocks) - 1
confidence_interval = t.interval(confidence_level, degrees_freedom, mean_difference, std_error)

print(f"Paired T-Test results: t-statistic = {t_statistic}, p-value = {p_value}")
print(f"Confidence interval: {confidence_interval}")

if p_value < 0.05:
    print("The difference between the trained model and the dummy model is statistically significant.")
else:
    print("The difference between the trained model and the dummy model is not statistically significant.")

# Paired T-Test results: t-statistic = 5.808870089050002, p-value = 1.6938206668063152e-06
# The difference between the trained model and the dummy model is statistically significant.
# Confidence interval: (898.0025827562124, 1866.115064302611)

import numpy as np
import matplotlib.pyplot as plt

# Your confidence interval
confidence_interval = (898.0025827562124, 1866.115064302611)

# Calculate the mean difference
mean_difference = np.mean(confidence_interval)

# Calculate the error (half the width of the confidence interval)
error = (confidence_interval[1] - confidence_interval[0]) / 2

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(1, mean_difference, yerr=error, fmt='o', color='b', ecolor='r', elinewidth=2, capsize=4)
plt.title('Confidence Interval for Mean Difference')
plt.xlabel('Sample')
plt.ylabel('Mean Difference')
plt.xlim(0, 2)  # Limit x-axis to focus on the error bar
plt.ylim(0, confidence_interval[1] + 200)  # Adjust y-axis to fit the error bar nicely
plt.grid(True)
plt.xticks([])  # Hide x-axis ticks as there's only one sample

# Add text to show the confidence interval values
plt.text(1, mean_difference + error + 50, f'CI: {confidence_interval}', ha='center', color='black')

plt.show()
