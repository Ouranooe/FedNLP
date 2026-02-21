import h5py
import os
path = os.path.expanduser("~/fednlp_data/partition_files/agnews_partition.h5")
with h5py.File(path, "r") as f:
    print(list(f.keys()))