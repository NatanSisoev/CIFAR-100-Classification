FILE_LIST = [
    "configs/config.py",

    "data/datasets.py",
    "data/transforms.py",
    "data/utils.py",

    "models/utils.py",
    "models/resnet01.py",

    "training/train01.py",
]

OUTPUT_FILE = "all_in_one.py"

# Prefixes of local modules to filter out from imports
LOCAL_MODULES = ["configs", "data", "models", "training", "."]


def should_keep_import(line):
    stripped = line.strip()
    if stripped.startswith("import "):
        for mod in LOCAL_MODULES:
            if stripped.split()[1].startswith(mod):
                return False
    elif stripped.startswith("from "):
        for mod in LOCAL_MODULES:
            if stripped.split()[1].startswith(mod):
                return False
    return True


with open(OUTPUT_FILE, "w") as out_file:
    for file_path in FILE_LIST:
        out_file.write(f"# ==== {file_path} ====\n")
        with open(file_path, "r") as f:
            for line in f:
                if should_keep_import(line):
                    out_file.write(line)
                else:
                    # skip local imports
                    continue
        out_file.write("\n\n")

print(f"All files combined into {OUTPUT_FILE}, local imports filtered out")
