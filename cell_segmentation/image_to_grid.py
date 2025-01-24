import numpy as np
from copy import deepcopy

def get_line_coordinates(x1, y1, x2, y2):
    """
    Calculate the pixel coordinates of a line between two points in a 2D space using Bresenham's line algorithm.

    This function computes the integer pixel coordinates for a straight line connecting
    two points `(x1, y1)` and `(x2, y2)`. It handles lines at any angle and ensures proper handling of steep lines
    and cases where the endpoints are swapped to always draw left to right.

    Args:
        x1 (int): X-coordinate of the starting point.
        y1 (int): Y-coordinate of the starting point.
        x2 (int): X-coordinate of the ending point.
        y2 (int): Y-coordinate of the ending point.

    Returns:
        list[tuple[int, int]]: A list of (x, y) tuples representing the pixel coordinates of the line.

    Example:
        >>> get_line_coordinates(0, 0, 3, 2)
        [(0, 0), (1, 0), (2, 1), (3, 2)]
    """
    # Calculate the differences in x and y coordinates.
    dx = abs(x2 - x1)  # The absolute difference in x-coordinates.
    dy = abs(y2 - y1)  # The absolute difference in y-coordinates.

    # Determine if the line is "steep", meaning it has a greater slope in the y-direction.
    steep = dy > dx

    if steep:
        # If the line is steep, we swap x and y for both points. This simplifies calculations
        # as we can treat a steep line as if it were less steep by rotating the grid.
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx  # Swap dx and dy to reflect the change in axis.

    if x1 > x2:
        # Ensure that we always draw from left to right. If x1 > x2, swap the points.
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # Initialize the error term to half of dx. This helps track when to increment y.
    error = dx / 2.0

    # Determine the direction to step in y (either +1 or -1) based on the relationship between y1 and y2.
    y_step = 1 if y1 < y2 else -1

    # Initialize y to the starting y-coordinate.
    y = y1

    # List to store the line's pixel coordinates.
    line_coordinates = []

    # Iterate over all x-coordinates from x1 to x2, inclusive.
    for x in range(x1, x2 + 1):
        # Append the current pixel to the line coordinates.
        # If the line was steep, the coordinates need to be flipped back.
        line_coordinates.append((y, x) if steep else (x, y))

        # Update the error term by subtracting dy.
        error -= dy

        if error < 0:
            # If the error term becomes negative, increment y (in the appropriate direction).
            y += y_step
            error += dx  # Reset the error term by adding dx.

    return line_coordinates

def find_cycles(graph, start, node, visited, path, cycles):
    """
    Recursive function to find all cycles starting and ending at a specific node.

    Args:
        graph (dict): Adjacency list representation of the graph.
        start (any): The starting node of the cycle.
        node (any): The current node being visited.
        visited (set): Set of visited nodes to prevent revisiting.
        path (list): Current path being traversed.
        cycles (list): List of detected cycles.

    Returns:
        None. Adds cycles to the `cycles` list.
    """
    # Mark the current node as visited and add it to the path
    visited.add(node)
    path.append(node)

    # Traverse all neighbors of the current node
    for neighbor in graph[node]:
        if neighbor == start:
            # A cycle is found if the neighbor is the start node
            cycles.append(path[:] + [start])  # Add a copy of the path to the cycles
        elif neighbor not in visited:
            # Continue exploring the neighbor
            find_cycles(graph, start, neighbor, visited, path, cycles)

    # Backtrack: remove the current node from visited and path
    visited.remove(node)
    path.pop()

def find_minimal_cycles(vertices, edges):
    """
    Finds all minimal cycles in a directed graph.

    Args:
        vertices (list): List of vertices in the graph.
        edges (list): List of edges in the graph, where each edge is a tuple (u, v).

    Returns:
        list: A list of minimal cycles, where each cycle is represented as a list of nodes.
    """
    # Build the graph as an adjacency list
    graph = {vertex: [] for vertex in vertices}

    for edge in edges:
        u, v = edge
        graph[u].append(v)  # Add the edge to the graph

    # List to store all detected cycles
    cycles = []
    for vertex in graph:
        visited = set()  # Track visited nodes for each starting vertex
        path = []  # Track the current traversal path
        find_cycles(graph, vertex, vertex, visited, path, cycles)

    # Filter out minimal cycles
    minimal_cycles = []
    for cycle in cycles:
        is_minimal = True
        for other_cycle in cycles:
            # A cycle is not minimal if it's a subset of another cycle
            if cycle != other_cycle and set(cycle).issubset(other_cycle):
                is_minimal = False
                break
        if is_minimal:
            minimal_cycles.append(cycle)

    return minimal_cycles

def make_cell_img(image, img_size, vor, cell_coord, color=(255, 0, 0, 255)):
    """
    Creates an image representation of a Voronoi diagram while filtering vertices and edges 
    based on specific conditions.

    Args:
        img_size (tuple): Size of the output image (height, width, channels).
        voronoi: Voronoi object (from scipy.spatial.Voronoi) containing Voronoi tessellation data.
        cell_coord: Coordinates of the cell centers for the Voronoi diagram.
        color (tuple): RGBA color value for the Voronoi edges.

    Returns:
        np.ndarray: A NumPy array representing the resulting Voronoi diagram as an image.
    """
    # Pixelize all Voronoi vertex coordinates
    vertice_int = list(
        map(lambda vertice: [int(vertice[0]), int(vertice[1])], map(np.round, vor.vertices))
    )

    # Create an array to track which vertices are within the image boundary
    is_in_img = np.ones(len(vertice_int))

    # Get image dimensions (assumes 3-channel image)
    max_y, max_x, _ = np.shape(image)
    print(f"max x is {max_x} and max y is {max_y}")

    # Mark vertices outside the image boundary as invalid
    for i, (x, y) in enumerate(vertice_int):
        if x < 0 or x >= max_x or y < 0 or y >= max_y:
            is_in_img[i] = 0

    # Initialize the final image (filled with white color)
    img = np.ones(img_size) * 255

    # Initialize a list for valid Voronoi regions
    full_regions = []

    # Collect the unique vertices that are inside the image boundary
    unique_vertices = []
    for i, bool_is_in_img in enumerate(is_in_img):
        if bool_is_in_img == 1:
            unique_vertices.append(i)

    # Create a dictionary to track the degree (number of connections) of each vertex
    vertice_deg = {}
    for ridge_vertice in vor.ridge_vertices:
        ver1, ver2 = ridge_vertice
        if ver1 in unique_vertices and ver2 in unique_vertices:
            vertice_deg.setdefault(ver1, []).append(ver2)
            vertice_deg.setdefault(ver2, []).append(ver1)

    # Iteratively remove leaf vertices (vertices with only one connection)
    has_leaves = True
    while has_leaves:
        has_leaves = False
        for is_leaf_vertice in list(vertice_deg.keys()):  # Copy keys to avoid iteration issues
            if len(vertice_deg[is_leaf_vertice]) == 1 and vertice_deg[is_leaf_vertice][0] != -1:
                vertice_deg[is_leaf_vertice] = [-1]  # Mark as removed
                for vertice in vertice_deg.keys():
                    if is_leaf_vertice in vertice_deg[vertice]:
                        vertice_deg[vertice].remove(is_leaf_vertice)
                    has_leaves = True

    # Draw the edges of the remaining Voronoi diagram onto the image
    for ridge_vertice in vor.ridge_vertices:
        ver1, ver2 = ridge_vertice
        if (
            ver1 in unique_vertices
            and ver2 in unique_vertices
            and len(vertice_deg[ver1]) >= 2
            and len(vertice_deg[ver2]) >= 2
        ):
            x1, y1 = vertice_int[ver1]
            x2, y2 = vertice_int[ver2]
            # Use the `get_line_coordinates` function to generate pixel coordinates for the edge
            line_coords = get_line_coordinates(x1, y1, x2, y2)
            for line_x, line_y in line_coords:
                img[line_y][line_x] = (0, 0, 0)  # Draw edge in black

    return img

# Function to stack arrays
def periodic_boundary_conditions(central_array: np.ndarray):
    """Stacks arrays to each side of the original array so that periodic boundary conditions are satisfied.
 
    ### Args:
        - `central_array (np.ndarray)`: central array to which copies of itself will be stacked
    """
    h_array = np.hstack([central_array]*3) # Horizontally stack array 3 times
    v_array = np.vstack([h_array]*3) # Then vertically stack this array 3 times
    return v_array

# Function to get periodic centroids
def get_periodic_centroids(centroids, w: int, h: int):
    """Computes periodic centroids of original centroids

    Args:
        centroids (_type_): list of lists, where items are x-y coordinates
        w (int): width
        h (int): height
    """
    # Initialize as periodic centroids
    periodic_centroids = deepcopy(centroids)
    
    # Loop over all widths and heights
    for width in [0, w, 2*w]:
        for height in [0, h, 2*h]:
            if (width == 0) and (height == 0): # Do nothing if width and height are 0 (equal to original)
                pass
            else:
                adjusted_centroids = [[el[0]+width, el[1]+height] for el in centroids]
                periodic_centroids.extend(adjusted_centroids)
        
    return periodic_centroids

# Function to get periodic contours
def get_periodic_contours(contours, w: int, h: int):
    """Computes periodic contours of original centroids

    Args:
        contours (_type_): list of lists, where items are x-y coordinates
        w (int): width
        h (int): height
    """
    # Initialize as periodic centroids
    periodic_contours = deepcopy(contours)
    
    # Loop over all widths and heights
    for width in [0, w, 2*w]:
        for height in [0, h, 2*h]:
            if (width == 0) and (height == 0): # Do nothing if width and height are 0 (equal to original)
                pass
            else:
                adjusted_contours = []
                for contour in contours:
                    contour_ = []
                    for xy in contour:
                        contour_.append([xy[0]+width, xy[1]+height])
                    adjusted_contours.append(contour_)
                periodic_contours.extend(adjusted_contours)
        
    return periodic_contours

# Function to get periodic bounding boxes
def get_periodic_bboxes(bboxes, w: int, h: int):
    """Computes periodic contours of original centroids

    Args:
        contours (_type_): list of lists, where items are x-y coordinates
        w (int): width
        h (int): height
    """
    # Initialize as periodic centroids
    periodic_bboxes = deepcopy(bboxes)
    
    # Loop over all widths and heights
    for width in [0, w, 2*w]:
        for height in [0, h, 2*h]:
            if (width == 0) and (height == 0): # Do nothing if width and height are 0 (equal to original)
                pass
            else:
                adjusted_bboxes = []
                for bbox in bboxes:
                    bbox_ = []
                    for xy in bbox:
                        bbox_.append([xy[0]+width, xy[1]+height])
                    adjusted_bboxes.append(bbox_)
                periodic_bboxes.extend(adjusted_bboxes)
        
    return periodic_bboxes