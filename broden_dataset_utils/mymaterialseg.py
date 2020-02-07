
import os
import csv
import numpy
import json
from scipy.misc import imread

from .loadseg import AbstractSegmentation

class MyMaterialSegmentation(AbstractSegmentation):

    def __init__(self, directory):
        """Constructor"""
        """ 
        Directory structure should be as follows.
        ${DIR}/${dir_for_each_data}/image.jpg
                                   /annotation.json
                                   /label.png
        """
        self.root_dir = directory

        self.info_list = []
        data_dirs = [ os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) \
                if os.path.isdir(os.path.join(self.root_dir, f)) ]
        for i, d in enumerate(data_dirs):
            with open(os.path.join(d, "annotation.json")) as f:
                data_meta = json.load(f)
                height = data_meta["imageHeight"]
                width = data_meta["imageWidth"]
            self.info_list.append({
                "dataset": "my_material",
                "file_index": i,
                "height": height,
                "width": width,
                "data_dir": d
            })
            self.material_dict = {}
            with open("./meta_file/joint_dataset/material.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    label = int(row[0])
                    name = row[1]
                    self.material_dict[label] = name

    def all_names(self, category, j):
        ## TODO what is this?????
        if j == 0:
            return []
        if category == 'material':
            return [ self.material_dict[j] ]
        return []

    def size(self, split=None):
        return len(self.info_list)

    def filename(self, i):
        return os.path.join(self.info_list[i]["data_dir"], "image.jpg")

    def metadata(self, i):
        """Returns an object that can be used to create all segmentations."""
        return dict(
            filename=os.path.join(self.info_list[i]["data_dir"], "image.jpg"),
            seg_filename=os.path.join(self.info_list[i]["data_dir"], "label.png"))

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        result = {}
        if wants('material', categories):
            labels = imread(m['seg_filename'])
            result['material'] = labels[:, :, 0]
        arrs = [a for a in list(result.values()) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

    def validation_dataset(self):
        """short-cut method to return validation dataset"""
        return self.info_list

def wants(what, option):
    if option is None:
        return True
    return what in option

