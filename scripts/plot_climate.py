import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("outputs/climate_results.csv")
loss=pd.read_csv("outputs/climate_loss.csv")

true_cols=[c for c in df.columns if c.startswith("true_")]
pred_cols=[c for c in df.columns if c.startswith("pred_")]
H=len(true_cols)

fig,axes=plt.subplots(1,3,figsize=(20,5))

zoom=500
start=max(0,len(df)//4)
sl=df.iloc[start:start+zoom]

for h in [0,H//2,H-1]:
    axes[0].plot(sl["idx"],sl[true_cols[h]],linewidth=1.2,label=f"True t+{h+1}",alpha=0.7)
    axes[0].plot(sl["idx"],sl[pred_cols[h]],linewidth=1.2,linestyle="--",label=f"Pred t+{h+1}",alpha=0.8)
axes[0].set_xlabel("Time (hours)")
axes[0].set_ylabel("Temperature (degC)")
axes[0].set_title(f"Multi-Step Forecast (H={H})")
axes[0].legend(fontsize=7)
axes[0].grid(True,alpha=0.3)

ax1=axes[1]
ax2=ax1.twinx()
ax1.plot(loss["Epoch"],loss["TrainLoss"],color="crimson",linewidth=2,label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss",color="crimson")
val=loss[loss["ValMAE"]>0]
ax2.plot(val["Epoch"],val["ValMAE"],color="teal",linewidth=2,marker='o',markersize=4,label="Val MAE")
ax2.set_ylabel("Val MAE (degC)",color="teal")
ax1.set_title("Training Loss & Val MAE")
ax1.legend(loc="upper right")
ax2.legend(loc="center right")
ax1.grid(True,alpha=0.3)

maes=[]
for h in range(H):
    err=np.abs(df[pred_cols[h]]-df[true_cols[h]])
    maes.append(err.mean())
axes[2].bar(range(1,H+1),maes,color=plt.cm.viridis(np.linspace(0.3,0.9,H)),edgecolor="black",linewidth=0.5)
axes[2].set_xlabel("Horizon (hours ahead)")
axes[2].set_ylabel("MAE (degC)")
axes[2].set_title("MAE by Forecast Horizon")
axes[2].set_xticks(range(1,H+1))
axes[2].grid(True,alpha=0.3,axis='y')

plt.tight_layout()
plt.savefig("outputs/climate_forecast_results.png",dpi=150)
plt.show()
print("Saved to outputs/climate_forecast_results.png")
