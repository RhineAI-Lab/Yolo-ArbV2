import os
from PIL import Image
import numpy as np

source_path = "E:\\dota"
target_path = "E:\\dota_yolo"

class_names = []

def cm_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

cm_dir(target_path + "\\images")
cm_dir(target_path + "\\labels")

for file in os.listdir(source_path + "\\labels"):
    print(file)
    results = []
    size = Image.open(source_path + "\\images\\" + file.split(".")[0] + ".png").size
    sn = np.tile(np.array(size),4)
    path = source_path + "\\labels\\" + file
    with open(path, "r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if i<2:
                continue
            v = line.split(" ")
            ps = np.array(list(map(float, v[:8])))
            psn = np.around(ps/sn,4)
            message = " ".join(list(map(str, psn.tolist())))
            try:
                i = class_names.index(v[8])
            except Exception:
                class_names.append(v[8])
                i = len(class_names)-1
            message = str(i) + " " + message
            results.append(message)
    with open(target_path + "\\labels\\" + file, "w") as f:
        f.write("\n".join(results))

print(class_names)







