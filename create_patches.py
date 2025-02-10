import numpy as np
import os

# Load nuclei and cells segmentations
nuclei_path = \
    r"/home/nixos/projects_2/cc3d-nucleus-modelling/nuclei_cell_segmentation/01_training_dataset_nuclei_npys/training_set_metastatic_roi_001.npy"
cells_path = \
    r"/home/nixos/projects_2/cc3d-nucleus-modelling/nuclei_cell_segmentation/01_training_dataset_cells_npys/training_set_metastatic_roi_001.npy"

nuclei = np.load(nuclei_path)
cells = np.load(cells_path)

# Crop
crop_idxs = np.linspace(0, nuclei.shape[0], 5, dtype=int)
nuclei_cropped = nuclei[
    crop_idxs[2]: crop_idxs[3],
    crop_idxs[2]: crop_idxs[3]
]
cells_cropped = cells[
    crop_idxs[2]: crop_idxs[3],
    crop_idxs[2]: crop_idxs[3]
]

# Save
submodule_path = r"/home/nixos/projects_2/cc3d-nucleus-modelling/nuclei_cell_segmentation"
np.save(
    os.path.join(submodule_path, "NUCLEI_CROPPED.npy"),
    nuclei
)
np.save(
    os.path.join(submodule_path, "CELLS_CROPPED.npy"),
    cells
)