"""In this notebook, all data will be stored in the respective folders."""

from packages import *
import pathlib
import nuclei_and_cell_segmentation as nc_seg

# Set some constants
IMAGES_DIR = r"01_training_dataset_tif_ROIs" # Images folder
IMAGES_LIST = os.listdir(IMAGES_DIR) # List of image file names
IMAGES_PATHS = [os.path.join(IMAGES_DIR, image) for image in IMAGES_LIST] # Paths to images

# Geojson files
GEOJSONS_DIR = r"01_training_dataset_geojson_nuclei"
GEOJSONS_LIST = os.listdir(GEOJSONS_DIR) 
GEOJSONS_PATHS = [os.path.join(GEOJSONS_DIR, image) for image in GEOJSONS_LIST] 

# Make dictionary for plots
PLOTS_DIR = r"01_training_dataset_segmentation_plots"
if not(os.path.exists(PLOTS_DIR)):
    os.mkdir(PLOTS_DIR)
    print("Folder for plots created!")
else:
    print("Folder for plots already exists ...")
    
# Save nuclei & cell segmentations as .npy file
# Specify whether to save them or not
save_nuclei = input("Save nuclei segmentations? [Y/N]\n")
save_cells = input("Save cell segmentations? [Y/N]\n")
    
# Make directory for nuclei segmentations
NUCLEI_NPYS_DIR = r"01_training_dataset_nuclei_npys"
if not(os.path.exists(NUCLEI_NPYS_DIR)):
    os.mkdir(NUCLEI_NPYS_DIR)
    print("Folder for nuclei .npy files created!")
else:
    print("Folder for nuclei .npy files already exists ...")
    
# Make directory for cell segmentations
CELLS_NPYS_DIR = r"01_training_dataset_cells_npys"
if not(os.path.exists(CELLS_NPYS_DIR)):
    os.mkdir(CELLS_NPYS_DIR)
    print("Folder for cells .npy files created!")
else:
    print("Folder for cells .npy files already exists ...")

# Specify whether to save nuclei props & cells props
save_nuclei_props = input("Save nuclei props? [Y/N]\n")
save_cell_props = input("Save cell props? [Y/N]\n")

# Make directory for nuclei props df
NUCLEI_PROPS_DIR = r"01_training_nuclei_props"
if not(os.path.exists(NUCLEI_PROPS_DIR)):
    os.mkdir(NUCLEI_PROPS_DIR)
    print("Folder for nuclei props created!")
else:
    print("Folder for nuclei props already exists ...")
    
# Make directory for nuclei props df
CELLS_PROPS_DIR = r"01_training_cells_props"
if not(os.path.exists(CELLS_PROPS_DIR)):
    os.mkdir(CELLS_PROPS_DIR)
    print("Folder for cells props created!")
else:
    print("Folder for cells props already exists ...")
    
# Store plots
for i, (image_path, geojson_path) in tqdm(enumerate(zip(IMAGES_PATHS, GEOJSONS_PATHS))):
    
    # Store file as pdf
    save_path = os.path.basename(
        image_path
    ).replace(".tif", ".pdf") # Got this from https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
    save_path = os.path.join(PLOTS_DIR, save_path)
    
    # Get segmentations
    labeled_nuclei_array, labeled_cells_array = nc_seg.tif_and_geojson_to_segmentation(
        image_path,
        geojson_path,
        show_progress=False # Do not show inner progress bar
    )
    
    # Save visualization
    nc_seg.plot_nuclei_and_cell_segmentations(
        labeled_nuclei_array, 
        labeled_cells_array, 
        save_path="", # Replace by actual path if you want to save plot (again)
        close=True # Do not plot visualizations within the loop
    )

    # Save nuclei segmentations if specified
    if save_nuclei.upper() == "Y":
        
        # Get path to store
        save_nuclei_path = os.path.basename(
            image_path
        ).replace(".tif", ".npy")
        save_nuclei_path = os.path.join(NUCLEI_NPYS_DIR, save_nuclei_path)
        
        # Save file
        with open(save_nuclei_path, "wb") as f:
            np.save(f, labeled_nuclei_array)
    
    # Save nuclei segmentations if specified
    if save_cells.upper() == "Y":
        
        # Get path to store
        save_cells_path = os.path.basename(
            image_path
        ).replace(".tif", ".npy")
        save_cells_path = os.path.join(CELLS_NPYS_DIR, save_cells_path)
        
        # Save file
        with open(save_cells_path, "wb") as f:
            np.save(f, labeled_cells_array)
            
    # Save nuclei props if specified
    if save_nuclei_props.upper() == "Y":
        
        # Get path to store
        save_nuclei_props_path = os.path.basename(
            image_path
        ).replace(".tif", ".xlsx")
        save_nuclei_props_path = os.path.join(NUCLEI_PROPS_DIR, save_nuclei_props_path)
        
        # Get props
        nuclei_props = \
            nc_seg.regionprops_from_mask(
                labeled_nuclei_array # DON'T MAKE THIS BINARY!!!
            )
        
        # Save file
        nuclei_props.to_excel(save_nuclei_props_path)
    
    # Save cell props if specified
    if save_cell_props.upper() == "Y":
        
        # Get path to store
        save_cell_props_path = os.path.basename(
            image_path,
        ).replace(".tif", ".xlsx")
        save_cell_props_path = os.path.join(CELLS_PROPS_DIR, save_cell_props_path)
        
        # Get props
        cell_props = \
            nc_seg.regionprops_from_mask(
                labeled_cells_array 
            )
        
        # Save file
        cell_props.to_excel(save_cell_props_path)