from .bases import BaseImageDataset
import os.path as osp
import glob
import re
import random
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import os
import math
import json


def save_json(dataset,json_path):
    dataset_as_list = [[img_path, pid, camid] for img_path, pid, camid in dataset]
    with open(json_path, 'w') as file:
        json.dump(dataset_as_list, file)

def load_json(json_path):
    with open(json_path, 'r') as file:
        loaded_dataset_as_list = json.load(file)
    loaded_dataset = [(img_path, pid, camid) for img_path, pid, camid in loaded_dataset_as_list]
    return loaded_dataset

def few_shot_setting(train_set, percent=0.5):
    pid_container = set()
    new_train_set = []
    for (img_path, pid, camid) in train_set:
        pid_container.add(pid)
    num_selected_pids = int(len(pid_container) * percent)
    selected_pids = set(random.sample(pid_container, num_selected_pids))
    for (img_path, pid, camid) in train_set:
        if pid in selected_pids:
            new_train_set.append((img_path, pid, camid))
    pid2label = {pid: label for label, pid in enumerate(selected_pids)}
    for idx, (img_path, pid, camid) in enumerate(new_train_set):
        if pid in selected_pids:
            new_train_set[idx] = (img_path, pid2label[pid], camid)
    return new_train_set


data_root = "/path/to/data/root"

class Market1501(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'Market-1501-v15.09.15'), verbose = True, few_shot_ratio=None, few_shot_seed=None):
        super(Market1501, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        if few_shot_seed:
            fewshot_path = os.path.join(self.dataset_dir,'fewshot_setting_{:.1f}_{}.json'.format(few_shot_ratio,few_shot_seed))
            if os.path.exists(fewshot_path):
                train = load_json(fewshot_path)
                print("Loaded {:.1%} IDs for training with seed {}.".format(few_shot_ratio,few_shot_seed))
            else:
                train = self._process_dir(self.train_dir, relabel=True)
                train = few_shot_setting(train_set=train,percent=few_shot_ratio)
                save_json(train,fewshot_path)
                print("Selected {:.1%} IDs for training.".format(few_shot_ratio))
        else:
            train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset



class DukeMTMCreID(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'DukeMTMC-reID'), verbose = True, few_shot_ratio=None, few_shot_seed=None):
        super(DukeMTMCreID, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        if few_shot_seed:
            fewshot_path = os.path.join(self.dataset_dir,'fewshot_setting_{:.1f}_{}.json'.format(few_shot_ratio,few_shot_seed))
            if os.path.exists(fewshot_path):
                train = load_json(fewshot_path)
                print("Loaded {:.1%} IDs for training with seed {}.".format(few_shot_ratio,few_shot_seed))
            else:
                train = self._process_dir(self.train_dir, relabel=True)
                train = few_shot_setting(train_set=train,percent=few_shot_ratio)
                save_json(train,fewshot_path)
                print("Selected {:.1%} IDs for training.".format(few_shot_ratio))
        else:
            train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMCreID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset




class CUHK03(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'CUHK03-NP'), verbose = True, few_shot_ratio=None, few_shot_seed=None):
        super(CUHK03, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'detected/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'detected/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'detected/bounding_box_test')

        if few_shot_seed:
            fewshot_path = os.path.join(self.dataset_dir,'fewshot_setting_{:.1f}_{}.json'.format(few_shot_ratio,few_shot_seed))
            if os.path.exists(fewshot_path):
                train = load_json(fewshot_path)
                print("Loaded {:.1%} IDs for training with seed {}.".format(few_shot_ratio,few_shot_seed))
            else:
                train = self._process_dir(self.train_dir, relabel=True)
                train = few_shot_setting(train_set=train,percent=few_shot_ratio)
                save_json(train,fewshot_path)
                print("Selected {:.1%} IDs for training.".format(few_shot_ratio))
        else:
            train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUHK03 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset



class CUHK03NP(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'CUHK03-NP'), verbose = True):
        super(CUHK03NP, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, '*/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, '*/query')
        self.gallery_dir = osp.join(self.dataset_dir, '*/bounding_box_test')

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUHK03NP loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset




class MSMT17_v1(BaseImageDataset):

    def __init__(self, data_dir = osp.join(data_root,'MSMT17_V1'), verbose = True, few_shot_ratio=None, few_shot_seed=None):
        super(MSMT17_v1, self).__init__()
        self.dataset_dir = data_dir

        self.train_dir = osp.join(self.dataset_dir, "train")
        self.test_dir = osp.join(self.dataset_dir, "test")
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')


        if few_shot_seed:
            fewshot_path = os.path.join(self.dataset_dir,'fewshot_setting_{:.1f}_{}.json'.format(few_shot_ratio,few_shot_seed))
            if os.path.exists(fewshot_path):
                train = load_json(fewshot_path)
                print("Loaded {:.1%} IDs for training with seed {}.".format(few_shot_ratio,few_shot_seed))
            else:
                train = self.process_dir(self.train_dir, self.list_train_path)
                train = few_shot_setting(train_set=train,percent=few_shot_ratio)
                save_json(train,fewshot_path)
                print("Selected {:.1%} IDs for training.".format(few_shot_ratio))
        else:
            train = self.process_dir(self.train_dir, self.list_train_path)
        # train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # # Note: to fairly compare with published methods on the conventional ReID setting,
        # #       do not add val images to the training set.
        # if 'combineall' in kwargs and kwargs['combineall']:
        #     train += val

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> MSMT17_v1 loaded")
            self.print_dataset_statistics(train, query, gallery)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        dataset = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))

        return dataset

class MSMT17(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'MSMT17'), verbose = True):
        super(MSMT17, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset




class ClonedPerson(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'ClonedPerson'), verbose=True):
        super(ClonedPerson, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/gallery')

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print('=> ClonedPerson loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_c([-\d]+)_f([-\d]+)')
        img_paths = sorted(glob.glob(osp.join(data_dir, '*g')))

        dataset = []
        all_pids = {}
        camera_offset = [0, 0, 0, 4, 4, 8, 12, 12, 12, 12, 16, 16, 20]

        for img_path in img_paths:
            fname = osp.basename(img_path)  # filename: id6_s2_c2_f6.jpg
            pid, scene, cam, _ = map(int, pattern.search(fname).groups())
            scene = 2
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            if relabel:
                pid = all_pids[pid]
            camid = camera_offset[scene] + cam
            dataset.append((img_path, pid, camid))
        return dataset



class RandPerson(BaseImageDataset):
    # 只有训练集 没有分gallery和query集
    def __init__(self, data_dir = osp.join(data_root,'RandPerson/images'), verbose=True):
        super(RandPerson, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, 'subset/randperson_subset/randperson_subset')

        train = self._process_dir(self.train_dir, relabel=True)

        if verbose:
            print('=> RandPerson loaded')
            self.print_dataset_statistics(train, train, train) #没有query和gallery但是函数必须接受三个输入

        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = sorted(glob.glob(osp.join(data_dir, '*g')))

        dataset = []
        all_pids = {}
        camera_offset = [0, 2, 4, 6, 8, 9, 10, 12, 13, 14, 15]

        for img_path in img_paths:
            fname = osp.basename(img_path)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            pid = int(fields[0])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            if relabel:
                pid = all_pids[pid]  # relabel
            camid = camera_offset[int(fields[1][1:])] + int(fields[2][1:])  # make it starting from 0
            dataset.append((img_path, pid, camid))
        return dataset



class PersonX(BaseImageDataset):
    def __init__(self, data_dir = osp.join(data_root,'PersonX_v1'), verbose = True):
        super(PersonX, self).__init__()
        self.dataset_dir = data_dir
        self.train_dir = osp.join(self.dataset_dir, '*/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, '*/query')
        self.gallery_dir = osp.join(self.dataset_dir, '*/bounding_box_test')

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PersonX loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*g'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset




class UnrealPerson(BaseImageDataset):
    def __init__(self, verbose=True):
        root = os.path.join(data_root,'UnrealPerson')
        dataset=['unreal_v1.1','unreal_v1.2','unreal_v1.3','unreal_v2.1','unreal_v2.2','unreal_v2.3','unreal_v3.1','unreal_v3.2','unreal_v3.3','unreal_v4.1','unreal_v4.2','unreal_v4.3']
        self.name = 'UnrealPerson'
        self.resampling = False
        self.root = root
        self.dataset_dir = dataset
        if type(self.dataset_dir)==list:
            self.dataset_dir = [osp.join(root,d) for d in self.dataset_dir]
            self.train_dir = [osp.join(d,'images') for d in self.dataset_dir]
        else:
            self.dataset_dir = osp.join(root, self.dataset_dir)
            self.train_dir = osp.join(self.dataset_dir, 'images')
        
        self.num_pids = 3000
        self.num_cams = 34
        self.img_per_person = 40
        # self.cams_of_dataset = None
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        
        list_train_file = os.path.join(root,'list_unreal_train.txt')
        if not os.path.exists(list_train_file):
            with open('list_unreal_train.txt','w') as train_file:
                for t in train:
                    train_file.write('{} {} {} \n'.format(t[0],t[1],t[2]))

        if verbose:
            print("=> UnrealPerson loaded")
            # self.print_dataset_statistics(train, query, gallery)
            self.print_dataset_statistics(train, train, train)

        self.train = train

        self.num_train_pids = num_train_pids

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def size(self,path):
        return os.path.getsize(path)/float(1024)

    def _process_dir(self, dir_path, relabel = True):
        list_train_file = os.path.join(self.root,'list_unreal_train.txt')
        if os.path.exists(list_train_file) and not self.resampling:
            dataset = []
            pid_container = set()
            with open(list_train_file, 'r') as file:
                for line in file:
                    entry = line.strip().split()
                    # (img_path, pid, camid) = (entry[0],int(entry[1]),int(entry[2]))
                    dataset.append((entry[0],int(entry[1]),int(entry[2])))
                    pid_container.add(int(entry[1]))
            num_pids = len(pid_container)
            num_imgs = len(dataset)
            return dataset, num_pids, num_imgs

        if type(dir_path)!=list:
            dir_path=[dir_path]

        cid_container = set()
        pid_container = set()

        img_paths =[]
        for d in dir_path:
            if not os.path.exists(d):
                assert False, 'Check unreal data dir'

            iii = glob.glob(osp.join(d, '*.*g'))
            print(d,len(iii))
            img_paths.extend( iii )
        
        # regex: scene id, person model version, person id, camera id ,frame id
        pattern = re.compile(r'unreal_v([\d]+).([\d]+)/images/([-\d]+)_c([\d]+)_([\d]+)')
        cid_container = set()
        pid_container = set()
        pid_container_sep = defaultdict(set)
        for img_path in img_paths:
            sid,pv, pid, cid,fid = map(int, pattern.search(img_path).groups())
            # if pv==3 and pid>=1800:
            #     continue
            # if pv<3 and pid>=1800: 
            #     continue # For training, we use 1600 models. Others may be used for testing later.
            cid_container.add((sid,cid))
            pid_container_sep[pv].add((pv,pid))
        for k in pid_container_sep.keys():
            print("Unreal pids ({}): {}".format(k,len(pid_container_sep[k])))
        print("Unreal cams: {}".format(len(cid_container)))
        # we need a balanced sampler here .
        num_pids_sep = self.num_pids // len(pid_container_sep)
        for k in pid_container_sep.keys():
            pid_container_sep[k]=random.sample(pid_container_sep[k],num_pids_sep) if len(pid_container_sep[k])>=num_pids_sep else pid_container_sep[k]
            for pid in pid_container_sep[k]:
                pid_container.add(pid)

        if self.num_cams!=0:
            cid_container = random.sample(cid_container,self.num_cams)
        print(cid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cid2label = {cid: label for label, cid in enumerate(cid_container)}
        if relabel == True:
            self.pid2label = pid2label
            self.cid2label = cid2label

        dataset = []
        img_per_id = defaultdict()
        img_per_id_per_cam = defaultdict(lambda: defaultdict(int))
        threshold_per_cam = math.ceil(self.img_per_person/self.num_cams)
        for img_path in tqdm(img_paths):
            if len(img_per_id)>=self.num_pids and len(dataset) > self.img_per_person*self.num_pids:
                # flag = True
                cnt = 0
                for pid in img_per_id.keys():
                    if len(img_per_id_per_cam[pid]) >= self.num_cams:
                        cnt += 1
                if cnt >= 100:
                    break
            sid, pv, pid, cid, fid = map(int, pattern.search(img_path).groups())
            if ((pv,pid) not in pid_container) or ((sid,cid) not in cid_container):
                continue
            if relabel:
                pid = pid2label[(pv,pid)]
                camid = cid2label[(sid,cid)]
            # if self.size(img_path)>2.5:
                if pid not in img_per_id.keys():
                    img_per_id[pid] = 1
                    img_per_id_per_cam[pid][camid] = 1
                    dataset.append((img_path, pid, camid))
                elif camid not in img_per_id_per_cam[pid].keys():
                    img_per_id[pid] += 1
                    img_per_id_per_cam[pid][camid] = 1
                    dataset.append((img_path, pid, camid))
                elif img_per_id[pid] >= (self.img_per_person+1) or img_per_id_per_cam[pid][camid]>=threshold_per_cam: #+1有个缓冲
                    continue
                else:
                    img_per_id[pid] += 1
                    img_per_id_per_cam[pid][camid] += 1
                    dataset.append((img_path, pid, camid))
        print("Sampled pids: {}".format(len(pid_container)))
        print("Sampled imgs: {}".format(len(dataset)))
        if relabel:
            self.pid2label = pid2label
            # if len(dataset)>self.img_per_person*self.num_pids:
            #     dataset=random.sample(dataset,self.img_per_person*self.num_pids)
        
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        print("Sampled imgs: {}".format(len(dataset)))
        return dataset, num_pids, num_imgs





class TrainDatasets(BaseImageDataset):
    def __init__(self, datasets = [], verbose = True):
        super(TrainDatasets, self).__init__()
        self.sub_datasets = datasets

        train = self._process_datasets(self.sub_datasets, relabel=True)

        if verbose:
            print("=> TrainDatasets loaded")
            self.print_dataset_statistics(train, train, train)

        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _process_datasets(self, sub_datasets, relabel=True):

        dataset = []
        pid_start = 0
        camid_start = 0
        for sub_dataset in sub_datasets:
            for (img_path, pid, camid) in sub_dataset.train:
                if relabel:
                    pid = pid_start+pid
                    camid = camid_start+camid
                dataset.append((img_path, pid, camid))
            num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(sub_dataset.train)
            pid_start += num_train_pids
            camid_start += num_train_cams

        return dataset



class TrainDatasetsBalanced(BaseImageDataset):
    def __init__(self, datasets = ['ClonedPerson','RandPerson','PersonX','UnrealPerson'], verbose = True):
        super(TrainDatasetsBalanced, self).__init__()
        __Datasets_Loaders = {
            "ClonedPerson": ClonedPerson,
            "RandPerson": RandPerson,
            "PersonX": PersonX,
            "UnrealPerson": UnrealPerson
        }
        self.sub_datasets = []
        for dataset in datasets:
            self.sub_datasets.append(__Datasets_Loaders[dataset](verbose=verbose))

        self.max_sub_size = 0
        self.min_sub_size = len(self.sub_datasets[0].train)
        
        for sub_dataset in self.sub_datasets:
            self.max_sub_size = max(len(sub_dataset.train),self.max_sub_size)
            self.min_sub_size = min(len(sub_dataset.train),self.min_sub_size)

        train = self._process_datasets(self.sub_datasets, relabel=True)

        if verbose:
            print("=> TrainDatasetsBalanced loaded")
            self.print_dataset_statistics(train, train, train)

        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _process_datasets(self, sub_datasets, relabel=True):

        dataset = []
        pid_start = 0
        camid_start = 0
        for sub_dataset in sub_datasets:
            num_train_pids, _, num_train_cams = self.get_imagedata_info(sub_dataset.train)

            for (img_path, pid, camid) in sub_dataset.train:
                if relabel:
                    pid = pid_start+pid
                    camid = camid_start+camid
                dataset.append((img_path, pid, camid))

            pid_start += num_train_pids
            camid_start += num_train_cams

            print(len(dataset))

        return dataset