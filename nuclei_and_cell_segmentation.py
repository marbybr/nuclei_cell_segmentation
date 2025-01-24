"""
Nuclei and Cell Segmentation Toolkit
====================================

This Python script provides functions for segmenting nuclei and cells from histological or microscopic images. 
It includes utilities for loading data, performing segmentation, visualizing results, and extracting 
quantitative features such as region properties. The main functionality is based on image processing techniques 
and concepts like Voronoi tessellation, watershed segmentation, and regionprops analysis.

Features:
---------
1. **Image Loading**: Load `.tif` images and GeoJSON annotation files for processing.
2. **Segmentation**: Perform nuclei and cell segmentation using binary masks and seeded watershed algorithms.
3. **Visualization**: Display results of segmentation, including labeled nuclei, cell boundaries, and bounding boxes.
4. **Feature Extraction**: Compute region properties for segmented nuclei or cells, including area, centroid, aspect ratio, and contours.
5. **Pipeline**: End-to-end functionality to process images and annotations into labeled segmentations.

Key Functions:
--------------
- `load_tif_image(path: str)`: Load a `.tif` image and return it as a NumPy array.
- `show_image(image: np.ndarray)`: Visualize an image with customizable size and axis display.
- `load_geojson(path_to_file: str)`: Load a GeoJSON annotation file and extract features.
- `regionprops_from_mask(binary_mask: np.ndarray)`: Extract region properties and contours from a binary mask.
- `segment_cells_from_nuclei(binary_mask: np.ndarray)`: Segment cells from nuclei boundaries using Voronoi-based algorithms.
- `plot_nuclei_and_cell_segmentations(binary_mask: np.ndarray, labeled_cells: np.ndarray)`: Plot labeled nuclei and cell boundaries with distinct colors.
- `tif_and_geojson_to_segmentation(tif_image_path: str, geojson_path: str)`: Complete pipeline from loading data to producing labeled segmentations.

Dependencies:
-------------
- **Python Libraries**: NumPy, Matplotlib, GeoJSON, scikit-image, pandas, tqdm
- **External Functions**: Voronoi-based labeling and seeded watershed from the `napari-segment-blobs-and-things-with-membranes` package.

Usage:
------
The script can be used for analyzing and visualizing cellular and nuclear structures in histological images. 
It is especially suited for datasets with annotated nuclei and cell boundaries.

Example:
--------
```python
# Load image and annotation data
image_path = "example.tif"
geojson_path = "annotations.geojson"
labeled_nuclei, labeled_cells = tif_and_geojson_to_segmentation(image_path, geojson_path)

# Visualize results
plot_nuclei_and_cell_segmentations(binary_mask=labeled_nuclei, labeled_cells=labeled_cells)
"""

# Import packages
from packages import *

# Function to load tif images as NumPy array
def load_tif_image(path: str):
    """Loads a .tif image from a given path and returns it as a NumPy array.
    Code from: https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy

    ### Args:
        - `path (str)`: path of the image
    """
    im = Image.open(path)
    imarray = np.array(im)
    return imarray

# Function to plot an image
def show_image(image: np.ndarray, figsize: tuple = (15, 8), axis: str = "off", title="ROI"):
    """## Visualizes an image

    ### Args:
        - `image (np.ndarray)`: NumPy array representing an image
        - `figsize (tuple, optional)`: width and height of the figure. Defaults to (15, 8).
        - `axis (str, optional)`: whether to include the axis ticks. Defaults to "off".
        - `title (str, optional)`: title of the image. Defaults to \"ROI\".
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis(axis)
    plt.title(title, fontsize=figsize[0])
    plt.show()
    
# Lambda function to convert 4d image to 3d
convert_4d_to_3d = lambda image: image[:, :, :-1]

# Function to show regionprops for an intensity image
def show_regionprops(image: np.ndarray, figsize: tuple = (15, 8), axis: str = "off", title="ROI"):
    """## Visualizes regionprops results on an image

    ### Args:
        - `image (np.ndarray)`: NumPy array representing an image
        - `figsize (tuple, optional)`: width and height of the figure. Defaults to (15, 8).
        - `axis (str, optional)`: whether to include the axis ticks. Defaults to "off".
        - `title (str, optional)`: title of the image. Defaults to \"ROI\".
    """
    # First convert image to grey scale
    if image.ndim == 3:
        image = rgb2gray(image)
    
    # Get edges
    edges = feature.canny(image, sigma=3, low_threshold=image.max()*0.2, high_threshold=image.max()*0.2)
    
    # Show regionprops results on image
    label_image = label(edges)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=plt.cm.gray)
    
    for region in regionprops(label_image):
    # Draw rectangle around segmented regions.
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    ax.set_title(title, fontsize=figsize[0])
    ax.axis(axis)
    plt.show()
    
# Function to load geojson file
def load_geojson(path_to_file: str):
    """Loads a geojson file and returns the features
    Code from: https://stackoverflow.com/questions/42753745/how-can-i-parse-geojson-with-python

    ### Args:
        - `path_to_file (str)`: path to GeoJSON file
    """
    with open(path_to_file) as f:
        gj = geojson.load(f)
    features = gj['features']
    return features

# Function to get coordinates
def get_coordinates_from_geometry(geojson_entry: dict):
    """Reads coordinates from a geojson file element

    ### Args:
        - `geojson_entry (dict)`: Entry of a geojson file
    """
    coordinates = geojson_entry["geometry"]["coordinates"]
    coordinates = np.asarray(coordinates).reshape(-1, 2).astype(np.int64)
    return np.asarray(coordinates)

# Lambda function to combine binary masks
def combine_binary_masks(*binary_masks):
    binary_mask = reduce(lambda bm1, bm2: np.logical_or(bm1, bm2), binary_masks)
    return binary_mask

# Function to get regionprops from individual masks
def regionprops_from_mask(
    binary_mask: np.ndarray,
    classification_name: str = "",
    start_label: int = 0) -> pd.DataFrame:
    """Computes regionprops for a binary mask.

    ### Args:
        - `binary_mask (np.ndarray)`: binary 2D NumPy array
        - `classification_name (str, optional)`: name of classification type, ignored if empty string. Defaults to "".
        - `start_label (int, optional)`: number to add to segmented label indices. Defaults to 0.

    ### Returns:
        - `pd.DataFrame`: table containing region properties extracted from binary mask
    """
    # Get connected components & regionprops
    label_img = label(binary_mask)
    
    # Get regionprops
    props = regionprops_table(
        label_img,
        properties=('label', 'area', 'centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'perimeter', 'coords'),
    )
    
    # Extract contours for each region
    contours = []
    
    for index in range(1, label_img.max()+1):
        idx = np.argwhere(props["label"] == index)[0][0]
        coords = props['coords'][idx]
        rr, cc = polygon(coords[:, 1], coords[:, 0])
        contours.append(np.asarray(list(zip(rr, cc))).astype(int).tolist())
    props['contours'] = contours
    
    # Add aspect_ratio & standardized cell shape index
    props["aspect_ratio"] = props["axis_major_length"] / props["axis_minor_length"]
    props["aspect_ratio_stand"] = \
        (props["aspect_ratio"] - props["aspect_ratio"].mean()) / props["aspect_ratio"].std()
    
    props["cell_shape_index"] = props["perimeter"] / np.sqrt(props["area"])
    props["cell_shape_index_stand"] = \
        (props["cell_shape_index"] - props["cell_shape_index"].mean()) / props["cell_shape_index"].std()
    
    # Add info about classification type if provided
    if not(classification_name == ""):
        props["classification_name"] = [classification_name] * len(props["label"])
    
    # Increment label if start_label > 0
    if start_label > 0:
        props["label"] = props["label"] + start_label
    
    return pd.DataFrame(props)

# Function to segment cell boundaries from nuclei boundaries
def segment_cells_from_nuclei(
    binary_mask: np.ndarray,
    spot_sigma: int = 0,
    outline_sigma: int = 4
):
    """Segments cells based on nuclei segmentations
    Code partly from: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/tree/main/napari_segment_blobs_and_things_with_membranes

    ### Args:
        - `binary_mask (np.ndarray)`: binary 2D NumPy array with segmented nuclei
        - `spot_sigma (int, optional)`: enhances regions with potential seed points using Gaussian smoothing. Defaults to 0.
        - `outline_sigma (int, optional)`: creates cleaner boundaries for the segmented regions using a Gaussian filter. \
            Defaults to 4.
    """
    # Get labeled nuclei
    labeled_nuclei = voronoi_otsu_labeling(binary_mask, spot_sigma=spot_sigma, outline_sigma=outline_sigma)
    
    # Get labeled cells
    labeled_cells = seeded_watershed(binary_mask, labeled_nuclei)
    
    # Return both labeled nuclei & labeled cells as NumPy arrays
    labeled_nuclei_array = np.asarray(labeled_nuclei.data)
    labeled_cells_array = np.asarray(labeled_cells.data)
    
    return labeled_nuclei_array, labeled_cells_array

# Function to perform total pipeline from image to grid
def tif_and_geojson_to_segmentation(
    tif_image_path: str,
    geojson_path: str,
    spot_sigma: int = 0,
    outline_sigma: int = 4,
    show_progress: bool = True
) -> np.ndarray:
    """Performs total pipeline from loading data to converting it to segmentations

    ### Args:
        - `tif_image_path (str)`: _description_
        - `geojson_path (str)`: _description_
        - `spot_sigma (int, optional)`: enhances regions with potential seed points using Gaussian smoothing. Defaults to 0.
        - `outline_sigma (int, optional)`: creates cleaner boundaries for the segmented regions using a Gaussian filter. \
            Defaults to 4.
        - show_progress (bool): If True, shows a progress bar. Defaults to True.

    ### Returns:
        - `np.ndarray`: NumPy array containing labeled nuclei array and labeled cells array
    """
    # Load data
    image = load_tif_image(tif_image_path)
    json_contents = load_geojson(geojson_path)
    
    # Create binary mask
    image_shape = image.shape # Image shape for binary mask
    image_shape = image_shape[:2] if len(image_shape) > 2 else image_shape # 2D Binary mask
    binary_mask = np.zeros(image_shape, dtype=np.int32)
    
    # Use tqdm if show_progress is True
    iterator = tqdm(json_contents) if show_progress else json_contents
    
    # Compute segmentations
    for entry in iterator:
        try:
            
            # Get coordinates & polygon from coordinates
            coords = get_coordinates_from_geometry(entry)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=image_shape)
            
            # Fill the contour mask with the polygon
            binary_mask[rr, cc] = 1
        except:
            pass
        
    # Then segment nuclei and labels
    labeled_nuclei_array, labeled_cells_array = segment_cells_from_nuclei(
        binary_mask, 
        spot_sigma, 
        outline_sigma)
    
    return labeled_nuclei_array, labeled_cells_array

# Function to plot segmentations
def plot_nuclei_and_cell_segmentations(
    binary_mask: np.ndarray, 
    labeled_cells: np.ndarray,
    figsize: tuple = (15, 8),
    save_path: str = "",
    title: str = "Cell & Nuclei Segmentation",
    close: bool = True
):
    """Plots the segmented nuclei and cells

    ### Args:
        - `binary_mask (np.ndarray)`: NumPy array containing segmented nuclei.
        - `labeled_cells (np.ndarray)`: NumPy array containing labeled cells.
        - `figsize (tuple, optional)`: width and height of the figure. Defaults to (15, 8).
        - `save_path (str, optional)`: path to save plot. If empty string, plot won't be saved. Defaults to \"\".
        - `title (str, optional)`: title of the image. Defaults to \"Cell & Nuclei Segmentation\".
        - `close (bool, optional)`: whether to show the plot (False) or not (True). Defaults to True.
    """
    # Find the boundaries of labeled cells
    cell_boundaries = find_boundaries(labeled_cells, mode='inner')

    # Plot the original binary mask
    plt.figure(figsize=figsize)
    plt.imshow(binary_mask, cmap="gray", alpha=0.5)  # Display the binary mask as background

    # Overlay the cell boundaries in red
    plt.imshow(cell_boundaries, cmap="Reds", alpha=0.7)

    # Optional: Add boundaries for nuclei (if needed)
    nuclei_boundaries = find_boundaries(binary_mask, mode='inner')
    plt.imshow(nuclei_boundaries, cmap="Blues", alpha=0.5)

    # Add title and remove axes
    plt.title(title, fontsize=figsize[0])
    
    # Save file if specified
    if save_path != "":
        plt.savefig(save_path, bbox_inches="tight")
        
    # Show plot if close is False
    if not(close):
        plt.show()
    else:
        plt.close()
    
# Function to show regionprops for segmentations
def show_regionprops_for_segmentation(
    segmentation: np.ndarray,
    figsize: tuple = (15, 8),
    axis: str = "off",
    title: str = "Segmentation bboxes"
):
    """Visualizes regionprops (i.e., bounding boxes) for segmentation

    ### Args:
        - `segmentation (np.ndarray)`: NumPy array containing segmentation
        - `figsize (tuple, optional)`: width and height of the figure. Defaults to (15, 8).
        - `axis (str, optional)`: whether to include the axis ticks. Defaults to "off".
        - `title (str, optional)`: title of the image. Defaults to \"Segmentation bboxes\".
    """
    # Label segmentation & get boundaries
    segmentation_regionprops = label(segmentation)
    segmentation_boundaries = find_boundaries(segmentation)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show boundaries
    ax.imshow(segmentation_boundaries, cmap=plt.cm.gray)

    # Draw rectangle around segmented regions.
    for region in regionprops(segmentation_regionprops):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    ax.set_title(title, fontsize=figsize[0])
    ax.axis("off")
    plt.show()