import json
import matplotlib.pyplot as plt

def get_loss(path):
    with open(path, 'r') as f:
        data = json.load(f)
    steps = []
    losses = []
    for log in data['log_history']:
        if 'loss' in log and 'step' in log:
            steps.append(log['step'])
            losses.append(log['loss'])
    return steps, losses

steps_08, loss_08 = get_loss('data/checkpoints_Qwen3.5-0.8B/checkpoint-55/trainer_state.json')
steps_2, loss_2 = get_loss('data/checkpoints_Qwen3.5-2B/checkpoint-55/trainer_state.json')

plt.figure(figsize=(8, 5))
plt.plot(steps_08, loss_08, label='Qwen3.5-0.8B', color='blue', alpha=0.7)
plt.plot(steps_2, loss_2, label='Qwen3.5-2B', color='red', alpha=0.7)

plt.title('Training Loss Curve (LoRA Fine-Tuning)')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('assignment-3/images/training_loss.png', dpi=300)
print("Saved to assignment-3/images/training_loss.png")
