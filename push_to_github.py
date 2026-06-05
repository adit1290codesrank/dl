import os
import subprocess
import glob

def clean_and_rename():
    print("\n--- 1. Cleaning & Renaming ---")
    directory = "examples"
    
    renames = {
        "alexa.cpp": "train_alexa.cpp",
        "cifar.cpp": "train_cifar.cpp",
        "emnist.cpp": "train_emnist.cpp",
        "eval.cpp": "eval_alexa.cpp",
        "test.cpp": "eval_cifar.cpp"
    }
    
    for old_name, new_name in renames.items():
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
            
    # Clean up massive .exe files and unnecessary SVGs
    for ext in ["*.exe", "*.svg"]:
        for file in glob.glob(ext):
            os.remove(file)
            print(f"Deleted unnecessary file: {file}")

def print_structure():
    print("\n--- 2. Final Repository Structure ---")
    for root, dirs, files in os.walk("."):
        if ".git" in root or "data" in root or "weights" in root or "__pycache__" in root or "build" in root:
            continue
            
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in sorted(files):
            # Only print source files and markdown
            if f.endswith(('.cpp', '.cu', '.h', '.md', 'Makefile')):
                print(f"{subindent}{f}")

def push_to_git():
    print("\n--- 3. Pushing to GitHub ---")
    
    # We must explicitly ignore data, weights, and binaries so we don't upload 1GB of data
    with open(".gitignore", "w") as f:
        f.write("weights/\ndata/\nbuild/\n*.exe\n*.bin\n*.o\n*.log\n")
    
    commands = [
        "git init",
        "git branch -M main",
        "git remote set-url origin https://github.com/adit1290codesrank/deeplearning.git 2>nul || git remote add origin https://github.com/adit1290codesrank/deeplearning.git",
        "git add -A",
        "git commit --amend -m \"Initial commit: Custom C++/CUDA Deep Learning Engine\" || git commit -m \"Initial commit: Custom C++/CUDA Deep Learning Engine\"",
        "git push -u origin main -f"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        # Run command and print output to terminal
        subprocess.run(cmd, shell=True)
    print("\n✅ Finished running git commands!")

if __name__ == "__main__":
    clean_and_rename()
    print_structure()
    
    val = input("\nDoes this structure look good? Do you want to push to GitHub now? (y/n): ")
    if val.lower() == 'y':
        push_to_git()
    else:
        print("Push cancelled.")
