import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('config.json')
# Load the TensorBoard event file
event_file = config['tensorboard_event_file']['event_file_path']

# Create an EventAccumulator object to parse the event file
event_acc = EventAccumulator(event_file)
event_acc.Reload()

# Extract the summary data
tags = event_acc.Tags()['scalars']
print("Available tags:", tags)

# Plot the metrics
plt.figure(figsize=(10, 5))

# Plot loss
if 'Loss/train' in tags:
    loss_train = event_acc.Scalars('Loss/train')
    loss_train_steps = [scalar.step for scalar in loss_train]
    loss_train_values = [scalar.value for scalar in loss_train]
    plt.plot(loss_train_steps, loss_train_values, label='Training Loss', marker='o')

if 'Loss/val' in tags:
    loss_val = event_acc.Scalars('Loss/val')
    loss_val_steps = [scalar.step for scalar in loss_val]
    loss_val_values = [scalar.value for scalar in loss_val]
    plt.plot(loss_val_steps, loss_val_values, label='Validation Loss', marker='o')

plt.title('Training and Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))

if 'Accuracy/val' in tags:
    acc_val = event_acc.Scalars('Accuracy/val')
    acc_val_steps = [scalar.step for scalar in acc_val]
    acc_val_values = [scalar.value for scalar in acc_val]
    plt.plot(acc_val_steps, acc_val_values, label='Validation Accuracy', marker='o')

plt.title('Validation Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()