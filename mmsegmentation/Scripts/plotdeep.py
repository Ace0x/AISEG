import json
import matplotlib.pyplot as plt

# Load the JSON log files
log_files = ['../work_dirs/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024/20240601_094409/vis_data/20240601_094409.json', '../work_dirs/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024/20240601_094409/vis_data/scalars.json']
log_data = []

for log_file in log_files:
    with open(log_file, 'r') as f:
        log_data.extend([json.loads(line) for line in f])

# Initialize lists to store metrics
iterations = []
loss = []
aAcc = []
mIoU = []

# Extract metrics from the log data
for entry in log_data:
    if 'iter' in entry:
        iterations.append(entry['iter'])
        loss.append(entry.get('loss', None))
        aAcc.append(entry.get('aAcc', None))
        mIoU.append(entry.get('mIoU', None))

# Remove None values and corresponding iterations
iterations, loss = zip(*[(i, l) for i, l in zip(iterations, loss) if l is not None])
iterations_aAcc, aAcc = zip(*[(i, a) for i, a in zip(iterations, aAcc) if a is not None])
iterations_mIoU, mIoU = zip(*[(i, m) for i, m in zip(iterations, mIoU) if m is not None])

# Plot the metrics
fig, ax1 = plt.subplots()

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(iterations, loss, label='Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Metrics', color='tab:red')
ax2.plot(iterations_aAcc, aAcc, label='aAcc', color='tab:green')
ax2.plot(iterations_mIoU, mIoU, label='meanIoU', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.title('Training Metrics Over Iterations')
plt.show()