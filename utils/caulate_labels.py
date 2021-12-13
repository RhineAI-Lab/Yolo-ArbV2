import numpy as np

inp = np.array([125,157,255,55,508,323,381,437])
size = np.array([640,480])
inp=inp.astype(np.float64)

bt = np.zeros([4])
box = np.zeros([4])
bt[0] = np.min(inp[::2])
bt[2] = np.max(inp[::2])
bt[1] = np.min(inp[1::2])
bt[3] = np.max(inp[1::2])
box[0] = (bt[0] + bt[2]) / 2
box[1] = (bt[1] + bt[3]) / 2
box[2] = bt[2] - bt[0]
box[3] = bt[3] - bt[1]

box[::2] /= size[0]
box[1::2] /= size[1]
inp[::2] /= size[0]
inp[1::2] /= size[1]

strbox = ""
for i in box:
    strbox += " "+format(i, '.4f')
strpoly = ""
for i in inp:
    strpoly += " "+format(i, '.4f')
print("0"+strbox)
print("0"+strpoly)
