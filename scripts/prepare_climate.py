import os,struct,zipfile,urllib.request,sys
import numpy as np

URL="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
ZIP_PATH="data/jena_climate.zip"
CSV_PATH="data/jena_climate_2009_2016.csv"
OUT_PATH="data/climate.bin"

WINDOW=24
HORIZON=6
SUBSAMPLE=6

FEATURE_COLS=["T (degC)","p (mbar)","rh (%)","VPdef (mbar)","wv (m/s)","rho (g/m**3)"]
TARGET_COL="T (degC)"

TRAIN_FRAC=0.70
VAL_FRAC=0.15

def download():
    os.makedirs("data",exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"[OK] CSV exists: {CSV_PATH}")
        return
    if not os.path.exists(ZIP_PATH):
        print("Downloading Jena Climate dataset ...")
        urllib.request.urlretrieve(URL,ZIP_PATH)
        print(f"[OK] Downloaded to {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH,'r') as zf:
        zf.extractall("data")
    print(f"[OK] Extracted to {CSV_PATH}")

def load_csv(path):
    with open(path,'r') as f:
        header=[h.strip('"').strip() for h in f.readline().strip().split(',')]
        rows=[]
        for line in f:
            parts=line.strip().split(',')
            row=[]
            for p in parts[1:]:
                try: row.append(float(p))
                except ValueError: row.append(0.0)
            rows.append(row)
    return header[1:],np.array(rows,dtype=np.float32)

def main():
    download()
    print("Loading CSV ...")
    cols,raw=load_csv(CSV_PATH)
    print(f"  Raw: {raw.shape}")

    data=raw[::SUBSAMPLE]
    print(f"  Hourly: {data.shape}")

    fi=[cols.index(c) for c in FEATURE_COLS]
    ti=0
    data_sel=data[:,fi]
    N,F=data_sel.shape
    print(f"  Features ({F}): {FEATURE_COLS}")

    wv_col=FEATURE_COLS.index("wv (m/s)")
    bad=np.sum(data_sel[:,wv_col]<0)
    if bad>0:
        print(f"  Cleaning {bad} negative wind values")
        data_sel[:,wv_col]=np.clip(data_sel[:,wv_col],0,None)

    n_tr=int(N*TRAIN_FRAC)
    n_va=int(N*(TRAIN_FRAC+VAL_FRAC))

    tr,va,te=data_sel[:n_tr],data_sel[n_tr:n_va],data_sel[n_va:]
    print(f"  Split: train={tr.shape[0]} val={va.shape[0]} test={te.shape[0]}")

    fmin,fmax=tr.min(axis=0),tr.max(axis=0)
    frange=fmax-fmin
    frange[frange==0]=1.0

    norm=lambda d:(d-fmin)/frange

    tr_n,va_n,te_n=norm(tr),norm(va),norm(te)

    y_min,y_max=float(fmin[ti]),float(fmax[ti])
    print(f"  T range: [{y_min:.2f}, {y_max:.2f}] degC")
    print(f"  Horizon: {HORIZON} steps")

    def make_windows(d):
        X,Y=[],[]
        for i in range(len(d)-WINDOW-HORIZON+1):
            X.append(d[i:i+WINDOW])
            Y.append(d[i+WINDOW:i+WINDOW+HORIZON,ti])
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)

    X_tr,Y_tr=make_windows(tr_n)
    X_va,Y_va=make_windows(va_n)
    X_te,Y_te=make_windows(te_n)

    print(f"  Windows: train={X_tr.shape} val={X_va.shape} test={X_te.shape}")
    print(f"  Targets: train={Y_tr.shape} val={Y_va.shape} test={Y_te.shape}")

    with open(OUT_PATH,'wb') as f:
        f.write(struct.pack('iiiiii',X_tr.shape[0],X_va.shape[0],X_te.shape[0],WINDOW,F,HORIZON))
        f.write(X_tr.tobytes());f.write(Y_tr.tobytes())
        f.write(X_va.tobytes());f.write(Y_va.tobytes())
        f.write(X_te.tobytes());f.write(Y_te.tobytes())
        f.write(struct.pack('ff',y_min,y_max))

    mb=os.path.getsize(OUT_PATH)/(1024*1024)
    print(f"\n[DONE] {OUT_PATH} ({mb:.1f} MB)")
    print(f"  W={WINDOW} H={HORIZON} F={F} train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]}")

if __name__=="__main__":
    main()
