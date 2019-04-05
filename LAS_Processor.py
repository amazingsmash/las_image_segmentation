from laspy.file import File
import numpy as np
import shutil
import os
import math
import gc
import imageio
from datetime import datetime

import easygui

from matplotlib import pyplot as plt



class LAS_Processor:

    def __init__(self, las_paths):
        self.__file_paths = las_paths

    def add_las_path(self, path):
        self.__file_paths += [path]

    def get_las_paths_in_folder(self, folder_path):
        self.__file_paths += [folder_path + "/" + f for f in os.listdir(folder_path) if ".las" in f]

    def split_two_longest_axis(xyzc, size):
        max_dim = np.argmax(size)
        m = np.median(xyzc[:, max_dim])
        division = xyzc[:, max_dim] > m
        return division

    def random_subsampling(xyzc, max_points):
        n_points = xyzc.shape[0]
        if n_points < max_points:
            return xyzc, np.array([0,4])

        selection = np.random.choice(n_points, size=max_points, replace=False)
        inverse_mask = np.ones(n_points, np.bool)
        inverse_mask[selection] = 0

        selection = xyzc[selection, :]
        non_selected = xyzc[inverse_mask, :]
        return selection, non_selected

    def aprox_average_distance(xyz):
        xyz0 = np.random.permutation(xyz)
        xyz1 = np.random.permutation(xyz)
        d = xyz1 - xyz0
        d = d[:,0]**2 + d[:,1]**2 + d[:,2]**2
        d = np.mean(d)
        return math.sqrt(d)

    def get_file_path(self, indices, out_folder):
        file_name = "Node"
        for i in indices: file_name += "_" + str(i)
        file_name += ".bytes"
        file_path = "%s/%s" % (out_folder, file_name)
        return file_name, file_path

    def generate_color_palette(point_classes):
        palette = sns.color_palette(None, len(point_classes))
        return [{"class": c, "color": list(palette[i])} for i, c in enumerate(point_classes)]

    def tileImage(img, res):
        if img.shape[0] < res[0] or img.shape[1] < res[1]:
            img[res[0], res[1]] = 0

        tiles = []
        for i in range(0, img.shape[0], res[0]):
            for j in range(0, img.shape[1], res[1]):
                tile = img[i:i+res[0], j:j+res[1]]
                tiles += [tile]

        return tiles

    def createZenithImages(xyzc, pixel_size, res, mask_class=16):

        xyz_offset = np.min(xyzc[:, 0:3], axis=0)
        xyzc[:, 0] -= xyz_offset[0]
        xyzc[:, 1] -= xyz_offset[1]
        xyzc[:, 2] -= xyz_offset[2]

        pixel = np.floor(xyzc[:,0:2] / pixel_size).astype(int)
        image_size = tuple(np.max(pixel, axis=0) + 1)
        image = np.zeros(image_size)
        mask = np.zeros(image_size)
        for i in range(xyzc.shape[0]):
            px = (pixel[i,0], pixel[i,1])
            oldz = image[px]
            image[px] = max([xyzc[i,2], oldz]) # Z
            mask[px] = mask[px] or xyzc[i,3] == mask_class

        # plt.imshow(mask, interpolation='nearest')
        # plt.show()
        img_tiles = LAS_Processor.tileImage(image, res)
        mask_tiles = LAS_Processor.tileImage(mask, res)

        return img_tiles, mask_tiles







    def generate(self, out_path="", max_las_files=None):

        out_folder = out_path
        shutil.rmtree(out_folder, ignore_errors=True)
        os.mkdir(out_folder)

        files = self.__file_paths
        if max_las_files is not None: files = files[:max_las_files]

        for index, file in enumerate(files):

            index_file_name = "tree_%d.json" % index

            print("Reading file %s" % file)
            in_file = File(file, mode='r')

            xyzc = np.transpose(np.array([in_file.x,
                                         in_file.y,
                                         in_file.z,
                                         in_file.Classification.astype(float)]))

            res = (256, 256)
            img_tiles, mask_tiles = LAS_Processor.createZenithImages(xyzc, pixel_size=0.2, res=res)

            min_pixels = (res[0] * res[1]) * 0.1
            for i in range(len(img_tiles)):
                img = img_tiles[i]
                if np.sum(img != 0) > min_pixels:
                    filename = "%s/image_%d_%d.png" % (out_folder, index, i)
                    imageio.imwrite(filename, img)
                    filename = "%s/mask_%d_%d.png" % (out_folder, index, i)
                    imageio.imwrite(filename, mask_tiles[i])

            gc.collect() # Forcing garbage collection

if __name__ == "__main__":

    #model = PointCloudModel("Corridor_New", [])
    # model.get_las_paths_in_folder(easygui.diropenbox("Select Data Folder"))
    #model.get_las_paths_in_folder("/Volumes/My Passport/Disco2/221_400BEG-PIE/LIDAR")

    model = LAS_Processor(["000029.las"])

    out_path = "Images/"
    # out_path = "/Volumes/My Passport/Unity_PC_Model/"

    t0 = datetime.now()
    model.generate(out_path)
    t1 = datetime.now()
    td = t1-t0

    print("Model Generated in %f sec." % td.total_seconds())