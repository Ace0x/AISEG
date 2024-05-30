#HERE
import os
import random

def send_to_correct(file, train_set=True):
    final_path = "train/" if train_set else "val/"
    if 'mask' in file:
        path = f'../data/land_cover/ann_dir/{final_path}' + file
        print(path)
        os.rename("train_resized/"+file,path)
    else:
        path = f'../data/land_cover/img_dir/{final_path}' + file
        print(path)
        os.rename("train_resized/"+file,f'../data/land_cover/img_dir/{final_path}' + file)

def separate_dataset():
    dict_map = {}

    if not os.path.exists('../data/land_cover'):
        os.makedirs('../data/land_cover')

    if not os.path.exists('../data/land_cover/ann_dir'):
        os.makedirs('../data/land_cover/ann_dir')
    
    if not os.path.exists('../data/land_cover/img_dir'):
        os.makedirs('../data/land_cover/img_dir')

    if not os.path.exists('../data/land_cover/ann_dir/train'):
        os.makedirs('../data/land_cover/ann_dir/train')
    
    if not os.path.exists('../data/land_cover/ann_dir/val'):
        os.makedirs('../data/land_cover/ann_dir/val')

    if not os.path.exists('../data/land_cover/img_dir/val'):
        os.makedirs('../data/land_cover/img_dir/val')

    if not os.path.exists('../data/land_cover/img_dir/train'):
        os.makedirs('../data/land_cover/img_dir/train')

    for i,file in enumerate(os.listdir('train_resized')):
        index = file.split("_")[0]
        if index in dict_map:
            rand_int = dict_map[index]
        else:
            rand_int = random.randint(0, 9)
            dict_map[index] = rand_int
        if rand_int < 8:
            send_to_correct(file)
        else:
            send_to_correct(file, False)

separate_dataset()