# Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io as sio
import json
from pathlib import Path
import image_to_grid as itog
from copy import deepcopy
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

# Path where input images are + image name
input_path = r"/home/nixos/projects_2/cc3d-nucleus-modelling/sample_images"
title = r"sample_image_2501"
name = title + ".png"

# Path where model segmentations are
output_path = r"/home/nixos/projects_2/cc3d-nucleus-modelling/sample_segmentations"

# Specify path to store sample images
PERIODIC_IMAGES_DIR = r"/home/nixos/projects_2/cc3d-nucleus-modelling/cell_to_grid_results/periodic_image_to_grid"

# define the tile and output paths
tile_path = Path(input_path, name)
tile_json_path = Path(output_path, "json", title+".json")
tile_mat_path = Path(output_path, "mat", title+".mat")
tile_overlay_path = Path(output_path,"overlay", name)

# 1) Tile processing output

# First let's view the 2D output from tile processing mode, that is stored in a .mat file.
image = plt.imread(tile_path)

# Dimensions of the original image
h, w = image.shape[:2]

# Define bounds for the central image in the periodic grid
x_start = w
x_end = 2 * w
y_start = h
y_end = 2 * h

# Image with periodic boundaries
periodic_image = cv2.cvtColor(itog.periodic_boundary_conditions(image), cv2.COLOR_BGR2RGB)

# Specify whether to show image or not
show_image = input("Show periodic image? [Y/N]\n")
if show_image.upper() == "Y":
    plt.figure()
    plt.imshow(periodic_image)
    # Add vertical and horizontal lines around the central image
    plt.axvline(x=x_start, color='red', linestyle='--', linewidth=2)  # Left boundary
    plt.axvline(x=x_end, color='red', linestyle='--', linewidth=2)    # Right boundary
    plt.axhline(y=y_start, color='red', linestyle='--', linewidth=2)  # Top boundary
    plt.axhline(y=y_end, color='red', linestyle='--', linewidth=2)    # Bottom boundary
    plt.axis("off") # No axis
    plt.title("Minimum Image Convention", fontsize=15) # Title of plot
    
    # Add arrows pointing outwards from the boundaries
    arrow_params = dict(color='blue', linewidth=2, headwidth=15)
    arrow_length = 150 # Arrow length
    arrow_start = 15 # Where arrow should start
    
    # Left arrow (pointing outwards)
    plt.annotate("", 
                xy=(x_start - arrow_length, (y_start + y_end) / 2),  # Arrowhead
                xytext=(x_start - arrow_start, (y_start + y_end) / 2),  # Arrow tail
                arrowprops=arrow_params)

    # Right arrow (pointing outwards)
    plt.annotate("", 
                xy=(x_end + arrow_length, (y_start + y_end) / 2),  # Arrowhead
                xytext=(x_end + arrow_start, (y_start + y_end) / 2),  # Arrow tail
                arrowprops=arrow_params)

    # Top arrow (pointing outwards)
    plt.annotate("", 
                xy=((x_start + x_end) / 2, y_start - arrow_length),  # Arrowhead
                xytext=((x_start + x_end) / 2, y_start - arrow_start),  # Arrow tail
                arrowprops=arrow_params)

    # Bottom arrow (pointing outwards)
    plt.annotate("", 
                xy=((x_start + x_end) / 2, y_end + arrow_length),  # Arrowhead
                xytext=((x_start + x_end) / 2, y_end + arrow_start),  # Arrow tail
                arrowprops=arrow_params)
        
    # Save figure if not already exists
    if not(
        os.path.exists(
            os.path.join(PERIODIC_IMAGES_DIR, f"{title}.pdf")
        )
    ):
        plt.savefig(os.path.join(PERIODIC_IMAGES_DIR, f"{title}.pdf"))
        print("Periodic image saved!")
    plt.show()

# Get the corresponding `.mat` file 
result_mat = sio.loadmat(tile_mat_path)

# Get the overlay
overlay = plt.imread(tile_overlay_path)
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) 

# Load the json file and add the contents to corresponding lists

bbox_list = []
centroid_list = []
contour_list = [] 
type_list = []


with open(tile_json_path) as json_file:
    data = json.load(json_file)
    mag_info = data['mag']
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        centroid_list.append(inst_centroid)
        inst_contour = inst_info['contour']
        contour_list.append(inst_contour)
        inst_bbox = inst_info['bbox']
        bbox_list.append(inst_bbox)
        inst_type = inst_info['type']
        type_list.append(inst_type)
        
# # get the number of items in each list

# print('Number of centroids', len(centroid_list))
# print('Number of contours', len(contour_list))
# print('Number of bounding boxes', len(bbox_list))

# # each item is a list of coordinates - let's take a look!
# print('-'*60)
# print(centroid_list[0])
# print('-'*60)
# print(contour_list[0])
# print('-'*60)
# print(bbox_list[0])     
    
# Get periodic elements
periodic_centroid_list = itog.get_periodic_centroids(centroid_list, w, h)
periodic_contour_list = itog.get_periodic_contours(contour_list, w, h)
periodic_bbox_list = itog.get_periodic_bboxes(bbox_list, w, h)

# # Get number of periodic elements
# print(len(periodic_centroid_list) / len(centroid_list) == 9)
# print(len(periodic_contour_list) / len(contour_list) == 9)
# print(len(periodic_bbox_list) / len(bbox_list) == 9)
# print(periodic_contour_list[0])
# print(periodic_contour_list[len(periodic_contour_list) - len(contour_list)])
# print(periodic_bbox_list[0])
# print(periodic_bbox_list[len(periodic_bbox_list) - len(bbox_list)])

# 2) Estimate the cell borders with voronoi tesselations (scipy)
from scipy.spatial import Voronoi, voronoi_plot_2d 

vor = Voronoi(periodic_centroid_list) 

# Make contours on the image
overlay = periodic_image.copy()
for contour in periodic_contour_list:
    overlay = cv2.drawContours(overlay.astype('float32'), [np.array(contour)], -1, (255,255,0), 1)

# Specify whether to show voronoi or not
show_voronoi = input("Show voronoi? [Y/N]\n")
if show_voronoi.upper() == "Y":
    fig = voronoi_plot_2d(vor, show_vertices=False)
    plt.imshow(overlay)
    
    # Add vertical and horizontal lines around the central image
    plt.axvline(x=x_start, color='red', linestyle='--', linewidth=2)  # Left boundary
    plt.axvline(x=x_end, color='red', linestyle='--', linewidth=2)  # Right boundary
    plt.axhline(y=y_start, color='red', linestyle='--', linewidth=2)  # Top boundary
    plt.axhline(y=y_end, color='red', linestyle='--', linewidth=2)  # Bottom boundary
    
    plt.axis('off')
    plt.title("Segmented Cells & Nuclei")
    
    # Save figure if not already exists
    if not(
        os.path.exists(
            os.path.join(PERIODIC_IMAGES_DIR, f"{title}_voronoi.pdf")
        )
    ):
        plt.savefig(os.path.join(PERIODIC_IMAGES_DIR, f"{title}_voronoi.pdf"))
        print("Periodic voronoi saved!")
    
    plt.show() 
    
from PIL import ImageDraw, Image
show_cell_grid = input("Show cell grid> [Y/N]\n")
if show_cell_grid.upper() == "Y":
    # Generate the Voronoi cell image using the `make_cell_img` function
    # `image` is the input image, `np.shape(image)` provides its dimensions
    # `vor` is the Voronoi tessellation object
    # Returns an image with Voronoi edges applied
    img = itog.make_cell_img(periodic_image, np.shape(periodic_image), vor, [])

    # Convert the generated NumPy array `img` to a PIL image for processing
    pil_img = Image.fromarray(np.uint8(img))

    # Perform flood-fill to fill the background of the Voronoi diagram
    # `xy=(0, 0)` specifies the starting pixel for the flood-fill operation (usually the top-left corner)
    # The `value=(255, 0, 0, 255)` specifies the color to fill the background (red in RGBA format)
    ImageDraw.floodfill(pil_img, xy=(0, 0), value=(255, 0, 0, 255))

    # Convert the PIL image back to a NumPy array for further manipulation
    img = np.asarray(pil_img)

    # Create a boolean mask to identify regions that match specific colors
    # `mask_cytoplasm` identifies all pixels where the color matches `[255, 0, 0]` (red background)
    mask_cytoplasm = np.all(img == [255, 0, 0], axis=2)

    # `mask_cells` identifies all pixels where the color matches `[255, 255, 255]` (white cells)
    mask_cells = np.all(img == [255, 255, 255], axis=2)

    # Make a copy of the image to modify its colors
    img_col = img.copy()

    # Replace the color of all pixels in `mask_cytoplasm` (red) with `[255, 255, 255]` (white)
    img_col[mask_cytoplasm] = [255, 255, 255]

    # Replace the color of all pixels in `mask_cells` (white) with `[0, 0, 255]` (blue)
    img_col[mask_cells] = [0, 0, 255]

    show_periodic_segmentation = input("Show periodic segmentation? [Y/N]\n")
    if show_periodic_segmentation.upper() == "Y":
        # Set up the figure and axis for visualization
        fig, ax = plt.subplots()

        # Plot the original input image with full opacity (`alpha=1`)
        ax.imshow(periodic_image, alpha=1)

        # Overlay the processed Voronoi image (`img_col`) with full opacity (`alpha=1`)
        # `origin='lower'` ensures the overlay image aligns correctly with the base image
        ax.imshow(img_col, alpha=1, origin='lower')
        
        # Add vertical and horizontal lines around the central image
        ax.axvline(x=x_start, color='red', linestyle='--', linewidth=2)  # Left boundary
        ax.axvline(x=x_end, color='red', linestyle='--', linewidth=2)  # Right boundary
        ax.axhline(y=y_start, color='red', linestyle='--', linewidth=2)  # Top boundary
        ax.axhline(y=y_end, color='red', linestyle='--', linewidth=2)  # Bottom boundary
        
        plt.axis("off")
        
        # Save figure if not already exists
        if not(
            os.path.exists(
                os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic_full.pdf")
            )
        ):
            plt.savefig(os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic_full.pdf"))
            print("Periodic segmentation (full) saved!")

        # Display the combined result
        plt.show()
    
    # Generate the Voronoi cell image using the `make_cell_img` function
    # `image` is the input image, `np.shape(image)` provides its dimensions
    # `vor` is the Voronoi tessellation object
    # Returns an image with Voronoi edges applied
    img = itog.make_cell_img(periodic_image, np.shape(periodic_image), vor, [])

    # Convert the generated NumPy array `img` to a PIL image for processing
    pil_img = Image.fromarray(np.uint8(img))

    # Perform flood-fill to fill the background of the Voronoi diagram
    # `xy=(0, 0)` specifies the starting pixel for the flood-fill operation (usually the top-left corner)
    # The `value=(255, 0, 0, 255)` specifies the color to fill the background (red in RGBA format)
    ImageDraw.floodfill(pil_img, xy=(0, 0), value=(255, 0, 0, 255))

    # Convert the PIL image back to a NumPy array for further manipulation
    img = np.asarray(pil_img)

    # Create a boolean mask to identify regions that match specific colors
    # `mask_cytoplasm` identifies all pixels where the color matches `[255, 0, 0]` (red background)
    mask_cytoplasm = np.all(img == [255, 0, 0], axis=2)

    # `mask_cells` identifies all pixels where the color matches `[255, 255, 255]` (white cells)
    mask_cells = np.all(img == [255, 255, 255], axis=2)

    # Make a copy of the image to modify its colors
    img_col = img.copy()

    # Replace the color of all pixels in `mask_cytoplasm` (red) with `[255, 255, 255]` (white)
    img_col[mask_cytoplasm] = [255, 255, 255]

    # Replace the color of all pixels in `mask_cells` (white) with `[0, 0, 255]` (blue)
    img_col[mask_cells] = [0, 0, 255]

    show_original_and_grid = input("Show original and grid image side by side? [Y/N]\n")
    
    if show_original_and_grid.upper() == "Y":
        
        # Set up the figure and axis for visualization
        fig, (ax1, ax) = plt.subplots(figsize=(24, 8), nrows=1, ncols=2)

        # Plot the original input image with full opacity (`alpha=1`)
        ax.imshow(periodic_image, alpha=1)

        # Overlay the processed Voronoi image (`img_col`) with full opacity (`alpha=1`)
        # `origin='lower'` ensures the overlay image aligns correctly with the base image
        ax.imshow(img_col, alpha=1, origin='lower')
        
        # Initialize a list to hold all patches (i.e., filled contours)
        contours = []
        adjusted_contours = []
        
        # Plot contours
        for i, centroid in enumerate(periodic_centroid_list):
            x_c, y_c = centroid
            # if (x_start <= x <= x_end) and (y_start <= y <= y_end):
            
            # Extract the contour for the current nucleus
            contour = np.array(periodic_contour_list[i])
            # contour = np.vstack([contour, contour[0, :]])
            
            # Make deep copy of contour for adjusted contour
            adjusted_contour = deepcopy(contour)
            
            # Adjust contour a bit so  that they are inside the cell boundaries
            adjusted_contour[:, 0] = (adjusted_contour[:, 0] + x_c) / 2
            adjusted_contour[:, 1] = (adjusted_contour[:, 1] + y_c) / 2
            
            # # Plot the contour using Line2D (matplotlib)
            # ax.plot(contour[:, 0], contour[:, 1], color="yellow", linewidth=0.5)
            
            # Add as a Polygon patch
            polygon = Polygon(contour, closed=True)
            contours.append(polygon)
            
            polygon = Polygon(adjusted_contour, closed=True)
            adjusted_contours.append(polygon)
            
        # Add all patches to the axis as a collection
        patch_collection = PatchCollection(contours, facecolor="red", edgecolor="black", linewidths=1, alpha=0.5)
        ax.add_collection(patch_collection)
        
        patch_collection = PatchCollection(adjusted_contours, facecolor="yellow", edgecolor="black", linewidths=1, alpha=0.7)
        ax.add_collection(patch_collection)
        
        # Plot centroids
        for i, centroid in enumerate(periodic_centroid_list):
            x_c, y_c = centroid
            ax.scatter(x_c, y_c, marker=".", color="black", edgecolor="black", s=20)
        
        # Add legend
        legend_handles = [
            Line2D([], [], color='black', marker='.', linestyle='None',
                            markersize=5, label='Centroid'),
            Patch(facecolor="red", edgecolor="black", label="Original Contour", alpha=0.5),
            Patch(facecolor="yellow", edgecolor="black", label="Adjusted Contour", alpha=0.7)
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9)
            
        # Set x and y limit
        ax.set_xlim([x_start, x_end])
        ax.set_ylim([y_start, y_end])
        
        # Show image in left plot
        ax1.imshow(cv2.cvtColor(image[::-1], cv2.COLOR_BGR2RGB))
        
        ax1.axis("off")
        ax.axis("off")
        
        # Set figure title and subplot titles
        fig.suptitle("Original Image & Lattice", fontsize=20)
        ax1.set_title("Original H&E Stained Image", fontsize=15)
        ax.set_title("CPM Input", fontsize=15)
        
        # Save figure if not already exists
        if not(
            os.path.exists(
                os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic_with_original.pdf")
            )
        ):
            plt.savefig(os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic_with_original.pdf"))
            print("Periodic segmentation saved!")

        # Display the combined result
        plt.show()
        
    else:
        # Set up the figure and axis for visualization
        fig, ax = plt.subplots(figsize=(24, 8))

        # Plot the original input image with full opacity (`alpha=1`)
        ax.imshow(periodic_image, alpha=1)

        # Overlay the processed Voronoi image (`img_col`) with full opacity (`alpha=1`)
        # `origin='lower'` ensures the overlay image aligns correctly with the base image
        ax.imshow(img_col, alpha=1, origin='lower')
        
        # Initialize a list to hold all patches (i.e., filled contours)
        contours = []
        adjusted_contours = []
        
        # Plot contours
        for i, centroid in enumerate(periodic_centroid_list):
            x_c, y_c = centroid
            # if (x_start <= x <= x_end) and (y_start <= y <= y_end):
            
            # Extract the contour for the current nucleus
            contour = np.array(periodic_contour_list[i])
            # contour = np.vstack([contour, contour[0, :]])
            
            # Make deep copy of contour for adjusted contour
            adjusted_contour = deepcopy(contour)
            
            # Adjust contour a bit so  that they are inside the cell boundaries
            adjusted_contour[:, 0] = (adjusted_contour[:, 0] + x_c) / 2
            adjusted_contour[:, 1] = (adjusted_contour[:, 1] + y_c) / 2
            
            # # Plot the contour using Line2D (matplotlib)
            # ax.plot(contour[:, 0], contour[:, 1], color="yellow", linewidth=0.5)
            
            # Add as a Polygon patch
            polygon = Polygon(contour, closed=True)
            contours.append(polygon)
            
            polygon = Polygon(adjusted_contour, closed=True)
            adjusted_contours.append(polygon)
            
        # Add all patches to the axis as a collection
        patch_collection = PatchCollection(contours, facecolor="red", edgecolor="black", linewidths=1, alpha=0.5)
        ax.add_collection(patch_collection)
        
        patch_collection = PatchCollection(adjusted_contours, facecolor="yellow", edgecolor="black", linewidths=1, alpha=0.7)
        ax.add_collection(patch_collection)
        
        # Plot centroids
        for i, centroid in enumerate(periodic_centroid_list):
            x_c, y_c = centroid
            ax.scatter(x_c, y_c, marker=".", color="black", edgecolor="black", s=20)
        
        # Add legend
        legend_handles = [
            Line2D([], [], color='black', marker='.', linestyle='None',
                            markersize=5, label='Centroid'),
            Patch(facecolor="red", edgecolor="black", label="Original Contour", alpha=0.5),
            Patch(facecolor="yellow", edgecolor="black", label="Adjusted Contour", alpha=0.7)
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9)
            
        # Set x and y limit
        ax.set_xlim([x_start, x_end])
        ax.set_ylim([y_start, y_end])
        
        ax.axis("off")
        
        # Save figure if not already exists
        if not(
            os.path.exists(
                os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic.pdf")
            )
        ):
            plt.savefig(os.path.join(PERIODIC_IMAGES_DIR, f"{title}_segmentation_periodic.pdf"))
            print("Periodic segmentation saved!")

        # Display the combined result
        plt.show()
