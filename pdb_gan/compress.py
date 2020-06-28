import os
import numpy as np
import logging
import rich
from rich.logging import RichHandler


log = logging.getLogger("rich")
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])


path = "data/compressed/"
if not os.path.exists(path):
    os.mkdir(path)


log.info("Compressing and saving data in 'npz' format...")
for dimension in ["28", "64"]:
    data = sorted([file for file in os.listdir("data/") if dimension in file])
    data = [np.load("data/" + file) for file in data]
    # print([array.dtype for array in data], [array.nbytes / 1e6 for array in data])
    np.savez_compressed(path + "{}x{}.npz".format(dimension, dimension), binary=data[0], gray=data[1], rgb=data[2], labels=np.load("data/labels.npy"))
