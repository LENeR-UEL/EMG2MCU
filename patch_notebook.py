import json, os, re, sys
SRC = r"Pipeline_AutoML_API.ipynb"
DST = r"Pipeline_AutoML_API_local.ipynb"
DATA_FILE = "11-08-2026_14-59-29.txt"
DATA_PATH = "."
LOCAL_OUT = os.path.abspath("./output").replace("\\", "/")
with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)
drive_path_subs = 0
code_cells_touched = 0
PATTERNS = [
    (r"/content/drive/MyDrive/Auto-ML Pipeline", LOCAL_OUT),
    (r"/content/drive/MyDrive/AutoML", LOCAL_OUT),
    (r"/content/drive/MyDrive", LOCAL_OUT),
    (r"/content/drive", LOCAL_OUT),
]
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    code_cells_touched += 1
    new_src = []
    for line in cell["source"]:
        new_line = line
        for pat, rep in PATTERNS:
            new_line2 = re.sub(pat, rep, new_line)
            if new_line2 != new_line:
                drive_path_subs += 1
                new_line = new_line2
        new_src.append(new_line)
    cell["source"] = new_src
patched_data_cfg_cells = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "data_file=" in src and "data_path=" in src and "DataConfig" in src:
        new_src = []
        for line in cell["source"]:
            nl = line
            nl = re.sub(r"data_file=.+", "data_file=" + repr(DATA_FILE) + ",", nl)
            nl = re.sub(r"data_path=.+", "data_path=" + repr(DATA_PATH) + ",", nl)
            new_src.append(nl)
        cell["source"] = new_src
        patched_data_cfg_cells.append(i)
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("code_cells_touched", code_cells_touched)
print("drive_path_subs", drive_path_subs)
print("patched_data_cfg_cells", patched_data_cfg_cells)
print("dst", DST)
print("local_out", LOCAL_OUT)
