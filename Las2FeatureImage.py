import numpy as np
import os
import tools
import os
import scipy.io as sio


# def generate_features(point_xyzic, old_pixel_value):
#
#     new_pixel_value = [max([point_xyzic[2], old_pixel_value[0]]),       #Max Z
#                        min([point_xyzic[2], old_pixel_value[1]]),       #Min Z
#                        max([point_xyzic[3], old_pixel_value[2]]),       #Max I
#                        min([point_xyzic[3], old_pixel_value[3]]),       #Min I
#                        np.sum([point_xyzic[3], old_pixel_value[4]]),    #Sum I
#                        old_pixel_value[5] + 1]                          #N Points
#
#     return new_pixel_value
#
# def generate_feature_from_points(ps_xyzic):
#
#     if len(ps_xyzic.shape) == 1:
#         new_pixel_value = [ps_xyzic[2],  ps_xyzic[2],  ps_xyzic[3], ps_xyzic[3], ps_xyzic[3]]
#     else:
#         new_pixel_value = [max(ps_xyzic[:, 2]),  #Max Z
#                            min(ps_xyzic[:, 2]),  #Min Z
#                            max(ps_xyzic[:, 3]),  #Max I
#                            min(ps_xyzic[:, 3]),  #Min I
#                            np.mean(ps_xyzic[:, 3])]  #Mean I
#
#     return new_pixel_value

def extract_features_from_points(image_points, maxx, maxy):

    image = np.zeros((maxx, maxy, 2))

    for px, points in image_points.items():
        ps_xyzic = np.array(points)

        image[px[0], px[1], :] = [max(ps_xyzic[:,2]) - min(ps_xyzic[:,2]),
                                  np.mean(ps_xyzic[:, 3])]
    print("Image created")
    return image

def get_zenith_feature_image(xyzic, pixel_size, mask_class=16):

    xyz_offset = np.min(xyzic[:, 0:3], axis=0)
    xyzic[:, 0] -= xyz_offset[0]
    xyzic[:, 1] -= xyz_offset[1]
    xyzic[:, 2] -= xyz_offset[2]

    pixel_index = np.floor(xyzic[:,0:2] / pixel_size).astype(int)
    image_size = np.max(pixel_index, axis=0) + 1

    n_features = 6
    feat_img = np.empty((image_size[0], image_size[1], n_features))
    feat_img[:,:,:] = np.nan
    mask = np.zeros(image_size)

    image = {}
    pxs = [tuple(px) for px in pixel_index]
    for px in pxs: image[px] = []

    for i in range(xyzic.shape[0]):
        px = pxs[i]
        image[px] += [xyzic[i,:]]
        mask[px] = mask[px] or xyzic[i,4] == mask_class

    feat_img = extract_features_from_points(image, max(pixel_index[:,0])+1, max(pixel_index[:,1]+1))

    return feat_img, mask, pixel_index


# def get_zenith_feature_image_2(xyzic, pixel_size, mask_class=16):
#     xyz_offset = np.min(xyzic[:, 0:3], axis=0)
#     xyzic[:, 0] -= xyz_offset[0]
#     xyzic[:, 1] -= xyz_offset[1]
#     xyzic[:, 2] -= xyz_offset[2]
#
#     pixel_index = np.floor(xyzic[:, 0:2] / pixel_size).astype(int)
#     image_size = np.max(pixel_index, axis=0) + 1
#     n_features = len(generate_features_from_points(ps_xyzic=xyzic[0, :]))
#     feat_img = np.zeros((image_size[0], image_size[1], n_features))
#     mask = np.zeros(image_size)
#
#     col = pixel_index[:,0]
#     for c in np.unique(col):
#         xyzic_col = xyzic[col == c, :]
#         row = pixel_index[col == c, 1]
#         for r in np.unique(row):
#             pixel_points = xyzic_col[row == r, :]
#             feat_img[c, r, :] = generate_features_from_points(ps_xyzic=pixel_points)
#
#
#     return feat_img, mask, pixel_index


if __name__ == "__main__":
    folder = "/Volumes/My Passport/Disco1/202_400ASC-PIE/02_Lidar"
    folder = "/Users/josemiguelsn/Desktop/repos/LASViewer/Data"
    files = [folder + "/" + f for f in os.listdir(folder) if f.endswith(".las")]
    files.sort()

    for f in files:
        xyzic = tools.read_las(f)
        print("Converting %s" % f)
        feat_img, mask, pixel_index = get_zenith_feature_image(xyzic, pixel_size=0.2, mask_class=16)
        fn = os.path.basename(f)
        sio.savemat("%s.mat" % fn, {"feat_img": feat_img,
                                    "mask": mask,
                                    "pixel_index": pixel_index})

        print("Done")