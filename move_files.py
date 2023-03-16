import os
import glob
import random

# absolute path to search all text files inside a specific folder
path1 = "data/train/*.jpg"
path2 = "data/train_masks/*.gif"
path3 = "data/validation_imgs/*.jpg"
path4 = "data/validation_masks/*.gif"
files1 = glob.glob(path1)
files2 = glob.glob(path2)
files3 = glob.glob(path3)
files4 = glob.glob(path4)
val_data_len = 600 

print(f"{len(files1)}, {len(files2)}")
print(f"{len(files3)}, {len(files4)}")

chosen_indicies = random.sample(range(len(files1)), val_data_len)
for ind in range(val_data_len):
    file = files1[ind]
    mask_file = file.replace(".jpg", "_mask.gif").replace("train", "train_masks")
    # print(f"{file}, {mask_file}")
    # print(f"{mask_file}, {file}")

    os.rename(file, file.replace('train', 'validation_imgs'))
    os.rename(mask_file, mask_file.replace('train_masks', 'validation_masks'))
    
