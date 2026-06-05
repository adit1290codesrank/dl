import os

def organize():
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
        elif os.path.exists(new_path):
            print(f"Already exists: {new_name}")
        else:
            print(f"File not found: {old_path}")

    # The 6 core files we want to keep front and center:
    core_files = {
        "train_alexa.cpp", "eval_alexa.cpp", 
        "train_cifar.cpp", "eval_cifar.cpp",
        "train_emnist.cpp", "eval_emnist.cpp",
        "cli.cpp" # Keeping the interactive CLI as a bonus
    }
    
    print("\n--- Current Examples Directory ---")
    for f in sorted(os.listdir(directory)):
        if f.endswith('.cpp'):
            if f in core_files:
                print(f"[CORE] {f}")
            else:
                print(f"[OTHER] {f} (You can delete this if you only want the 6 core files)")

if __name__ == "__main__":
    organize()
