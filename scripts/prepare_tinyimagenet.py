import os,struct,zipfile,urllib.request,sys
import numpy as np
from PIL import Image

URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_PATH="data/tiny-imagenet-200.zip"
DIR="data/tiny-imagenet-200"
OUT_PATH="data/tinyimagenet.bin"
IMG_SIZE=64
NUM_CLASSES=200

def download():
    os.makedirs("data",exist_ok=True)
    if os.path.exists(DIR):
        print(f"[OK] Already extracted: {DIR}")
        return
    if not os.path.exists(ZIP_PATH):
        print("Downloading Tiny ImageNet (~237MB) ...")
        urllib.request.urlretrieve(URL,ZIP_PATH)
        print(f"[OK] Downloaded to {ZIP_PATH}")
    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH,'r') as zf:
        zf.extractall("data")
    print(f"[OK] Extracted to {DIR}")

def load_image(path):
    img=Image.open(path).convert("RGB")
    return np.array(img,dtype=np.uint8)

def main():
    download()

    wnids=open(os.path.join(DIR,"wnids.txt")).read().strip().split("\n")
    class_to_idx={w:i for i,w in enumerate(wnids)}

    print("Loading training images ...")
    X_train,Y_train=[],[]
    for i, cls_name in enumerate(wnids):
        if i % 10 == 0:
            print(f"  Processed {i}/200 classes...", end='\r')
        cls_dir=os.path.join(DIR,"train",cls_name,"images")
        if not os.path.exists(cls_dir):continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".JPEG"):continue
            img=load_image(os.path.join(cls_dir,fname))
            if img.shape!=(64,64,3):
                img=np.array(Image.fromarray(img).resize((64,64)),dtype=np.uint8)
            X_train.append(img)
            Y_train.append(class_to_idx[cls_name])
    X_train=np.array(X_train,dtype=np.uint8)
    Y_train=np.array(Y_train,dtype=np.uint8)
    print(f"  Train: {X_train.shape} labels: {Y_train.shape}")

    print("Loading validation images ...")
    val_ann={}
    with open(os.path.join(DIR,"val","val_annotations.txt")) as f:
        for line in f:
            parts=line.strip().split("\t")
            val_ann[parts[0]]=parts[1]

    X_val,Y_val=[],[]
    val_dir=os.path.join(DIR,"val","images")
    for fname in sorted(os.listdir(val_dir)):
        if not fname.endswith(".JPEG"):continue
        img=load_image(os.path.join(val_dir,fname))
        if img.shape!=(64,64,3):
            img=np.array(Image.fromarray(img).resize((64,64)),dtype=np.uint8)
        X_val.append(img)
        Y_val.append(class_to_idx[val_ann[fname]])
    X_val=np.array(X_val,dtype=np.uint8)
    Y_val=np.array(Y_val,dtype=np.uint8)
    print(f"  Val: {X_val.shape} labels: {Y_val.shape}")

    print("Saving binary ...")
    with open(OUT_PATH,'wb') as f:
        f.write(struct.pack('iiii',len(X_train),len(X_val),IMG_SIZE,NUM_CLASSES))
        f.write(X_train.tobytes())
        f.write(Y_train.tobytes())
        f.write(X_val.tobytes())
        f.write(Y_val.tobytes())

    mb=os.path.getsize(OUT_PATH)/(1024*1024)
    print(f"\n[DONE] {OUT_PATH} ({mb:.1f} MB)")
    print(f"  train={len(X_train)} val={len(X_val)} img={IMG_SIZE} classes={NUM_CLASSES}")

if __name__=="__main__":
    main()
