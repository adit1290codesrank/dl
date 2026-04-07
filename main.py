import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    df = pd.read_csv('./data/curve.csv')
except FileNotFoundError:
    print("Error: curve.csv not found.")
    exit()

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

plt.plot(df['x'], df['y_true'], label='True: sin(x)', color='#00ff00', linewidth=2, linestyle='--')
plt.plot(df['x'], df['y_pred'], label='Model Prediction', color='#ff00ff', alpha=0.8, linewidth=2)

# Mark the boundaries of the training data
plt.axvline(x=-2*np.pi, color='white', linestyle=':', alpha=0.5, label='Training Bounds')
plt.axvline(x=2*np.pi, color='white', linestyle=':', alpha=0.5)

plt.title('CUDA Engine: sin(x) with Save/Load Verification', fontsize=14)
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()