import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df = pd.read_csv("outputs/sin_results.csv")
axes[0].plot(df["x_last"], df["y_true_next"], label="Ground Truth", linewidth=2)
axes[0].plot(df["x_last"], df["y_pred_next"], label="Prediction", linewidth=2, linestyle="--")
axes[0].set_xlabel("x (last input)")
axes[0].set_ylabel("sin(x_next)")
axes[0].set_title("Transformer: sin(x)cos(2x) + 0.5sin(3x)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

loss = pd.read_csv("outputs/sine_transformer_loss.csv")
axes[1].plot(loss["Epoch"], loss["Loss"], label="Loss", color="crimson", linewidth=2)
ax2 = axes[1].twinx()
ax2.plot(loss["Epoch"], loss["LR"], label="LR", color="gray", linewidth=1, linestyle=":")
ax2.set_ylabel("Learning Rate", color="gray")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Training Loss & LR Schedule")
axes[1].legend(loc="upper right")
ax2.legend(loc="center right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/sine_transformer_results.png", dpi=150)
plt.show()
