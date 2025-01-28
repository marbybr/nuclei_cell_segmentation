import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import seaborn as sns
import skimage
from PIL import Image

# Load a cell segmentation
cell_seg = np.load(r"nuclei_cell_segmentation/01_training_dataset_cells_npys/training_set_metastatic_roi_001.npy")

# Compute boundaries
cell_boundaries = skimage.segmentation.find_boundaries(cell_seg).astype(np.int64)

# Define colormap
cMap = []
cMap.append((0, "blue"))
cMap.append((1, "black"))
cMap = LinearSegmentedColormap.from_list("custom", cMap)

# Show image
plt.figure()
plt.imshow(
    cell_boundaries,
    cmap = cMap
)
plt.axis("off")
plt.tight_layout()
plt.savefig("development_data_import_cc3d/cell_segmentation_0.png", bbox_inches="tight")
plt.show()