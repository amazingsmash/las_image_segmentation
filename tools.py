import shutil

from laspy.file import File
import numpy as np
import os
import imageio

import JSONUtils
from unet import readImagesFromFolder, unet
import skimage.io as io

from matplotlib import pyplot as plt

import skimage.transform as trans

def join_tiles_from_image_folder(folder, index):

    files = [folder + "/" + f for f in os.listdir(folder) if f.endswith(".png")]
    files.sort()

    images = [io.imread(fn, as_gray=True) for fn in files]
    image_names = [os.path.basename(f) for f in files]
    final_img = join_tiles(images, image_names, index)

    imageio.imwrite("%s/../predicted.png" % folder, final_img)

def join_tiles(images, image_names, index, res=(256,256)):

    img_size = (max([i["x"] + res[0] for i in index]),
                max([i["y"] + res[1] for i in index]))

    final_img = np.zeros((img_size[0], img_size[1]), dtype=float)

    for i in range(len(image_names)):
        fn = image_names[i]
        ii = [x for x in index if x["imageName"] == fn][0]
        x = ii["x"]
        y = ii["y"]
        final_img[x:x + res[0], y:y + res[1]] = images[i]

    return final_img

def calculate_probability(xyz, prob_image, pixel_size):
    xyz_offset = np.min(xyz, axis=0)
    xyz[:, 0] -= xyz_offset[0]
    xyz[:, 1] -= xyz_offset[1]
    xyz[:, 2] -= xyz_offset[2]

    pixel = np.floor(xyz[:, 0:2] / pixel_size).astype(int)
    print("Getting probabilities from image: %d x %d" % (max(pixel[:,0]),max(pixel[:,1])))
    prob = prob_image[pixel[:,0], pixel[:,1]]
    return prob

def predict_tiles(index, path='Images/000029.las'):
    model = unet(pretrained_weights="unet_turret.hdf5", input_size=(256, 256, 1))

    res = (256, 256)
    x, all_files = read_images_from_folder(path, 'images', thresholding=False, target_size=res, maxNImages=0,
                                           showImages=False)

    predicted_tiles = []
    file_names = []
    for i in range(0, x.shape[0], 3):
        images = x[i:i + 3]
        files = all_files[i:i + 3]

        p = model.predict(images)

        for pr in p: predicted_tiles += [pr.reshape(res)]
        for f in files: file_names += [os.path.basename(f)]

    return join_tiles(predicted_tiles, file_names, index, res=(256,256))

def read_las(file):
    print("Reading file %s" % file)
    in_file = File(file, mode='r')
    xyzc = np.transpose(np.array([in_file.x,
                                  in_file.y,
                                  in_file.z,
                                  in_file.Classification.astype(float)]))
    return xyzc

def get_zenith_tiles(xyzc, pixel_size, res, mask_class=16):

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
    img_tiles, tile_coord = tile_image(image, res)
    mask_tiles, tile_coord = tile_image(mask, res)
    return img_tiles, mask_tiles, tile_coord

def tile_image(img, res):
    img_shape_0 = np.math.ceil(img.shape[0] / res[0]) * res[0]
    img_shape_1 = np.math.ceil(img.shape[1] / res[1]) * res[1]
    img = np.pad(img,
                 ((0, img_shape_0 - img.shape[0]), (0, img_shape_1 - img.shape[1])),
                 'constant', constant_values=(0, 0))
    print("Creting tile for %d x %d image" % img.shape)
    tiles = []
    tile_coord = []
    for i in range(0, img.shape[0], res[0]):
        for j in range(0, img.shape[1], res[1]):
            tile = img[i:i+res[0], j:j+res[1]]
            tiles += [tile]
            tile_coord += [{"x": i, "y": j}]

    return tiles, tile_coord

def save_tiles_folder(xyzc, file, out_path=""):

    out_folder = out_path
    shutil.rmtree(out_folder, ignore_errors=True)
    os.mkdir(out_folder)

    res = (256, 256)
    img_tiles, mask_tiles, tile_coord = get_zenith_tiles(xyzc, pixel_size=0.2, res=res)

    path = "%s/%s" % (out_folder, file)
    JSONUtils.makeDirectory(path + "/images/")
    JSONUtils.makeDirectory(path + "/masks/")
    min_pixels = (res[0] * res[1]) * 0.01
    for i in range(len(img_tiles)):
        img = img_tiles[i]
        if np.sum(img != 0) > min_pixels:
            image_name = "%s_%d.png" % (file, i)
            mask_name = "%s_mask_%d.png" % (file, i)
            filename = "%s/images/%s" % (path, image_name)
            imageio.imwrite(filename, img)
            filename = "%s/masks/%s.png" % (path, mask_name)
            imageio.imwrite(filename, mask_tiles[i])
            tile_coord[i]["imageName"] = image_name
            tile_coord[i]["number"] = i

    tile_coord = [t for t in tile_coord if "imageName" in t]
    JSONUtils.writeJSON(tile_coord, "%s/images_index.json" % path)

    return tile_coord



def read_images_from_folder(path, folder, thresholding, target_size=(256, 256), maxNImages=0, showImages=False):
    path_x = path + "/" + folder
    files = [path_x + "/" + f for f in os.listdir(path_x) if f.endswith(".png")]
    files.sort()
    if maxNImages > 0: files = files[0:maxNImages]

    x = np.zeros((len(files), target_size[0], target_size[1], 1), dtype=float)
    # print(x.shape)
    for i, file in enumerate(files):
        # print(file)
        img = io.imread(file, as_gray=True)

        img = (255 - img) / 255 if not thresholding else img < 0.5

        img = trans.resize(img, (*target_size, 1))
        x[i] = img

        if showImages:
            print(img.dtype)
            plt.imshow(img[:, :, 0], interpolation='nearest')
            plt.show()

    return x, files
