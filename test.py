import numpy as np
import imageio
import JSONUtils
import tools

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.io as sio
from numpy import logical_and as l_and
from numpy import logical_not as l_not

import skimage.io as io

file = "000029.las"
xyzic = tools.read_las(file)
xyzc = xyzic[:, [0,1,2,4]]

if False:
    index = tools.save_tiles_folder(xyzc, file, out_path="Images/")
else:
    index = JSONUtils.readJSON(filename="Images/000029.las/images_index.json")

if False:
    final_img = tools.predict_tiles(index, path='Images/000029.las')
    imageio.imwrite("Images/000029.las/predicted_2.png", final_img)
    ps = tools.calculate_probability(xyzc[:,0:3], final_img, pixel_size=0.2)
    print(ps.shape)
    sio.savemat("results.mat", {"predictions": ps})

results = sio.loadmat("results.mat")
predictions = (255.0 - results["predictions"]) / 255.0

y_true = np.where(xyzc[:,3] == 16, True, False)
y_pred = np.where(predictions > 0.5, True, False).reshape(y_true.shape)
turret_points = np.nonzero(y_pred)[0]
turret_xyz = xyzc[turret_points, 0:3]

hits = y_pred == y_true
print("Accuracy: %f" % (np.count_nonzero(hits) / len(y_true)))
print("False Negative Rate: %f" % (np.sum(l_and(y_true, l_not(y_pred))) / len(y_true)))
print("False Positive Rate: %f" % (np.sum(l_and(l_not(y_true), y_pred)) / len(y_true)))
#plot_confusion_matrix(y_true, y_pred, ["No turret", "Turret"])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(turret_xyz[:,0], turret_xyz[:,1], turret_xyz[:,2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

sio.savemat("points.mat", {"predictions": predictions,
                                  "xyz": xyzic[:, [0,1,2]],
                                  "intensity": xyzic[:, 3],
                                  "class": xyzic[:, 4]})

plt.show()

print("DONE")

