import os


def return_jester(modality, root_dir):
    filename_categories = 'datasets/jester/category.txt'
    filename_imglist_train = 'datasets/jester/train_videofolder.txt'
    filename_imglist_val = 'datasets/jester/val_videofolder.txt'
    if modality == 'RGB' or modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = os.path.join(root_dir, 'datasets/jester')
    else:
        print('no such modality:'+modality)
        os._exit(0)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_nvgesture(modality, root_dir):
    filename_categories = 'datasets/nvgesture/category.txt'
    filename_imglist_train = 'datasets/nvgesture/train_videofolder.txt'
    filename_imglist_val = 'datasets/nvgesture/val_videofolder.txt'
    if modality == 'RGB' or modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = os.path.join(root_dir, 'datasets/nvgesture')
    else:
        print('no such modality:'+modality)
        os._exit(0)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_chalearn(modality, root_dir):
    filename_categories = 'datasets/chalearn/category.txt'
    filename_imglist_train = 'datasets/chalearn/train_videofolder.txt'
    filename_imglist_val = 'datasets/chalearn/val_videofolder.txt'
    if modality == 'RGB' or modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = os.path.join(root_dir, 'datasets/chalearn')
    else:
        print('no such modality:'+modality)
        os._exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(root_dir, dataset, modality):
    dict_single = {'jester':return_jester, 'nvgesture': return_nvgesture, 'chalearn': return_chalearn}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality, root_dir)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(root_dir, file_imglist_train)
    file_imglist_val = os.path.join(root_dir, file_imglist_val)
    file_categories = os.path.join(root_dir, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix

