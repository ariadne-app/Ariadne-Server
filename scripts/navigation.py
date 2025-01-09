import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rtree import index
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
from scipy.interpolate import make_interp_spline, splprep, splev
import pickle
from typing import Tuple, List, Literal, Callable
import json
from inference_sdk import InferenceHTTPClient
from functions import *
from tqdm import tqdm
import tempfile

MatLike = np.ndarray


class Indoor_Navigation:
    # Class variables
    name: str = None
    icon = None
    image_original = None
    image = None
    image_upscale = None
    walls = None
    doors = None
    walls_doors = None
    pixel_to_cm = None
    graph: nx.Graph = None
    graph_nodes = None
    contours: STRtree = None
    rooms = None
    # TODO: Deprecate this
    scale = 4

    _debug_ = False

    def __init__(self,
                 image_path: str,
                 icon_path: str = None,
                 name: str = None,
                 debug = False):
        """
        A class that represents an indoor navigation system
        """
        self.name = name
        # TODO: Load the icon
        self.image_original = cv2.imread(image_path)
        self._debug_ = debug
        # if self._debug_:
        #     self.plot_image(self.image_original)
    
    def calibrate(self, pixel_to_cm: float):
        """
        A function that calibrates the pixel to meters ratio
        """
        self.pixel_to_cm = pixel_to_cm
    
    def process_image(self, grid_size=30, radius=2.3, doors=None, rooms=None):
        """
        A function that processes the image
        """
        pbar = tqdm(total=4, desc="Overall Progress", unit="task")

        pbar.write("Initial image process...")
        self.image = clean_image(self.image_original)
        # if self._debug_:
        #     self.plot_image(self.image)
        # TODO: Implement the upscale factor based on the pixel_to_cm ratio
        # Darken black pixels
        filters = [threshold(70, 255, 'tozero')
                   ]
        image = apply_filters(self.image, filters)
        self.image_upscale = upscale_image(image)
        # if self._debug_:
        #     self.plot_image(self.image_upscale)
        self.image_upscale = sharpen_image(self.image_upscale, kernel=3)
        # if self._debug_:
        #     self.plot_image(self.image_upscale)
        pbar.update(1)
        pbar.write("Detecting walls...")
        self.walls = self.detect_walls(self.image_upscale, self.pixel_to_cm)
        # if self._debug_:
        #     self.plot_image(self.walls)
        pbar.update(1)
        pbar.write("Detecting doors...")
        self.doors, self.walls, self.walls_doors = self.detect_doors(doors=doors)
        # if self._debug_:
        #     self.plot_image(self.walls)
        # if self._debug_:
        #     self.plot_image(self.walls_doors)
        pbar.update(1)
        pbar.write("Creating Graph...")
        self.graph, self.graph_nodes, self.contours = self.generate_graph(grid_size=grid_size, radius=radius, rooms=rooms)
        pbar.update(1)
        pbar.close()

    def calculate_route(self,
                        start: Tuple[float, float],
                        end: Tuple[float, float],
                        algorithm: Literal['dijkstra', 'astar'] = 'astar',
                        in_pixels: bool = False,
                        simplify_route=True
                        ) -> Tuple[List[Tuple[float, float]], float]:
        """
        A function that calculates the shortest path between two points
        
        Parameters
        ----------
        start : The start point as a tuple (x, y)
        end : The end point as a tuple (x, y)
        algorithm : The algorithm to use for path calculation. Either 'dijkstra' or 'astar'
        in_pixels : If True, the start and end points are in pixels.
            If False, the start and end points are in relative coordinates (from 0 to 1).
            Also, the output path will be in relative coordinates if in_pixels is False.

        Returns
        -------
        path_points : List of tuples representing the path points
        distance : The total distance of the path in meters        
        """

        if not in_pixels:
            image_shape = self.image_upscale.shape
            start_pixel = (int(start[0] * image_shape[1]), int(start[1] * image_shape[0]))
            end_pixel = (int(end[0] * image_shape[1]), int(end[1] * image_shape[0]))

        # Find the nearest nodes to the start and end points
        start_node = min(self.graph_nodes, key=lambda x: np.sqrt((x[0] - start_pixel[0])**2 + (x[1] - start_pixel[1])**2))
        end_node = min(self.graph_nodes, key=lambda x: np.sqrt((x[0] - end_pixel[0])**2 + (x[1] - end_pixel[1])**2))

        # Find the shortest path
        if algorithm == 'dijkstra':
            path = nx.shortest_path(self.graph, source=self.graph_nodes.index(start_node), target=self.graph_nodes.index(end_node), weight='weight')
        elif algorithm == 'astar':
            # Heuristic function for A* (Euclidean distance)
            def heuristic(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
                x1, y1 = self.graph.nodes[node1]['pos']
                x2, y2 = self.graph.nodes[node2]['pos']
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            # Find the shortest path using A* algorithm
            path = nx.astar_path(self.graph, source=self.graph_nodes.index(start_node), target=self.graph_nodes.index(end_node), heuristic=heuristic, weight='weight')
        else:
            raise ValueError('Invalid algorithm. Choose either "dijkstra" or "astar"')
        
        # Calculate the total weight of the path
        distance = sum(self.graph[u][v]['weight'] for u, v in zip(path, path[1:]))

        path_points = [self.graph_nodes[i] for i in path]


        # add starting and ending nodes to the path
        if in_pixels:
            start_point = start
            end_point = end
        else:
            start_point = (start[0] * image_shape[1], start[1] * image_shape[0])
            end_point = (end[0] * image_shape[1], end[1] * image_shape[0])

        # Check if start and end points can be connected directly to the path
        # TODO: Check if self.image_upscale[start_point[1]][start_point[0]] is ok or we should swap the indices ?? deprecated->abort
        start_line = LineString([start_point, path_points[0]])
        end_line = LineString([end_point, path_points[-1]])

        if not any(True for _ in self.contours.query(start_line, 'intersects')):
            path_points.insert(0, start_point)
            print('Start point connected to the path')
        else:
            print('Start point --NOT-- connected to the path')
        if not any(True for _ in self.contours.query(end_line, 'intersects')):
            path_points.append(end_point)
            print('End point connected to the path')
        else:
            print('End point --NOT-- connected to the path')
        
        if simplify_route:
            # Try skipping path points. Check if removing a point the path will pass through a contour
            i = 0
            j = len(path_points) - 1
            z = 0
            
            while i < j - 1:
                if z % 2 == 0:
                    # Check from the start
                    line = LineString([path_points[i], path_points[i + 2]])
                    if not any(True for _ in self.contours.query(line, 'intersects')):
                        path_points.pop(i + 1)
                        j -= 1
                    else:
                        i += 1
                else:
                    # Check from the end
                    line = LineString([path_points[j], path_points[j - 2]])
                    if not any(True for _ in self.contours.query(line, 'intersects')):
                        path_points.pop(j - 1)
                        j -= 1
                    else:
                        j -= 1
                
                z += 1
            
            i = 0
            while i < len(path_points) - 2:
                line = LineString([path_points[i], path_points[i + 2]])
                if not any(True for _ in self.contours.query(line, 'intersects')):
                    path_points.pop(i+1)
                else:
                    i += 1

        if not in_pixels:
            path_points = [(x / image_shape[1], y / image_shape[0]) for(x, y) in path_points]

        # Calculate the total distance of the path in meters by summing the the euclidean distances between the points
        distance = sum([np.linalg.norm(np.array(i) - np.array(j)) for i, j in zip(path_points, path_points[1:])])

        return path_points, self.pixels_to_cm(distance, scale=4)

    def save(self, path: str) -> None:
        """A function that saves the object to a file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def predict_rooms(self, confidence=0.7):
        return None
    
    def post_rooms_labeled(self, rooms):
        self.rooms = rooms

    def get_json(self, save_path: str = None) -> None:
        """
        A function that returns the JSON representation of the object
        or saves it to a file if save_path is provided
        """

        # TODO: Implement primary nodes, labels, rooms, floor
        data = { node : {"edges" : [], "label" : self.graph.nodes[node].get('label'), "coords" : self.graph_nodes[node], "primary" : True} for node in self.graph.nodes }

        for i, j in self.graph.edges:
            data[i]["edges"].append(j)
            data[j]["edges"].append(i)

        json_graph = []

        # get items of dictionary
        for i, j in data.items():
            dict_i = {
                "id" : i,
                "edges" : j["edges"],
                "label" : j["label"],
                "imageX" : j["coords"][0]/self.scale,
                "imageY" : j["coords"][1]/self.scale,
                "primary" : j["primary"],
                "floor" : 0
            }
            json_graph.append(dict_i)
        
        # TODO: Implement room labels
        json_rooms = [
            {
                "label": label,
                "floor": 0,
                "coords": [{"x": x/self.scale, "y": y/self.scale} for x, y in room.exterior.coords]
            }
            for label, room in self.rooms.items()
        ]

        json_data = {
            "Distance_Per_Pixel" : self.pixel_to_cm,
            "Graph" : json_graph,
            "Rooms" : json_rooms,
            "Floors" : 1
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        else:
            return json_data

    @classmethod
    def load(self, path: str) -> None:
        """A function that loads the object from a file"""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
        
    def report(self) -> str:
        """A function to report image resolution, graph nodes, and graph edges, etc"""
        return (
            f"Report for Indoor Navigation Object:\n"
            f"-Name: {self.name}\n"
            f"-Image resolution: {self.image.shape}\n"
            f"-Number of graph nodes: {len(self.graph_nodes)}\n"
            f"-Number of graph edges: {len(self.graph.edges)}"
        )
    
    def plot_route(self, path_points: List[Tuple[float, float]], distance: float, in_pixels: bool = False) -> None:
        """A function that plots the path on the floor plan"""
        if not in_pixels:
            image_shape = self.image.shape
            path_points = [(int(point[0] * image_shape[1]), int(point[1] * image_shape[0])) for point in path_points]
        plt.imshow(self.image, cmap='gray')
        # Connect lines between the points
        for i in range(len(path_points) - 1):
            x = [path_points[i][0], path_points[i + 1][0]]
            y = [path_points[i][1], path_points[i + 1][1]]
            plt.plot(x, y, 'b-', markersize=1)
        for i in path_points:
            plt.plot(i[0], i[1], 'ro', markersize=2)
        plt.title('Shortest Path on Floor Plan\nDistance: {:.2f} meters'.format(distance))
        plt.show()
    
    def calculate_and_plot_route(self,
                                 start: Tuple[float, float],
                                 end: Tuple[float, float],
                                 algorithm: Literal['dijkstra', 'astar'] = 'astar',
                                 in_pixels: bool = False,
                                 simplify_route=True) -> None:
        """A function that calculates and plots the shortest path between two points"""
        path_points, distance = self.calculate_route(start, end, algorithm, in_pixels, simplify_route)
        self.plot_route(path_points, distance, in_pixels)
    
    def predict_class(self, classes, confidence=0.7):
        """
        A function that predicts the class of an image using a trained model
        """
        CLIENT = InferenceHTTPClient(
            api_url="https://outline.roboflow.com",
            api_key="mQL45QHB1lZ19v2mKuf1"
        )

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            image_path = temp_file.name
            cv2.imwrite(image_path, self.image_upscale)

            # predictions = CLIENT.infer(image_path, model_id="full-set-menu/5")['predictions']
            predictions = CLIENT.infer(image_path, model_id="doors-windows-detection/4")['predictions']

        # Filter the predictions to get the classes
        classes = [p for p in predictions if p['class'] in classes and p['confidence'] > confidence]

        return classes

    def detect_doors(self, doors=None, confidence=0.7):
        """
        A function that detects doors in the image
        This function returns the centers of the doors, an image of walls with doors erased, and an image of doors erased with doors closed
        """
        if doors is None:
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                image_path = temp_file.name
                cv2.imwrite(image_path, self.image_upscale)
                predictions = model_predict(image_path)
                classes = ['DOOR-SINGLE', 'DOOR-DOUBLE']
                doors = [p for p in predictions if p['class'] in classes and p['confidence'] > confidence]
        else:
            doors = [{'x': door['x'] * self.scale, 'y': door['y'] * self.scale, 'width': door['width'] * self.scale, 'height': door['height'] * self.scale} for door in doors]

        image_doors_erased = self.walls.copy()
        # TODO: Implemet door closing
        image_doors = self.walls.copy()
        door_centers = []

        factor = 1.3
        for i, door in enumerate(doors):
            x, y, width, height = door['x'], door['y'], door['width'], door['height']

            x1 = int(x - width * factor / 2)
            y1 = int(y - height * factor / 2)
            x2 = int(x + width * factor / 2)
            y2 = int(y + height * factor / 2)

            # Set coordinates to the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_doors_erased.shape[1], x2)
            y2 = min(image_doors_erased.shape[0], y2)

            image_doors_erased[y1:y2, x1:x2] = 255

            # Add the door center to the list
            door_centers.append((x, y))
        
        return door_centers, image_doors_erased, image_doors
        

    # TODO: Implement the pixel_to_cm
    def detect_walls(self, image, pixel_to_cm):
        # Apply gaussian blur and thresholding
        filters = [
            gaussian_blur((5,5), 0),
            threshold(240, 255)
            ]
        return apply_filters(image, filters)
    
    # TODO: this works for certain image dimensions. make it robust to upscale and downscale images
    def plot_image(self, image: MatLike, with_rooms=False, with_graph=False, save_file=None) -> None:
        """A function that plots the image"""
        if with_rooms:
            rooms, _ = cv2.findContours(self.walls_doors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # reject small contours
            rooms = [room for room in rooms if cv2.contourArea(room) > 500]
            rooms = [room for room in rooms if cv2.contourArea(room) < 10000000]

            # reject contours that circulate black areas
            def is_black_area(contour, image):
                # create a mask with the contour
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                # erode the mask
                mask = cv2.erode(mask, kernel(5), iterations=1)
                # a random black point of mask
                point = np.where(mask == 255)
                # check if the point is black
                return image[point[0][0], point[1][0]] == 0


            rooms = [room for room in rooms if not is_black_area(room, self.walls_doors)]    

            print('Number of rooms:', len(rooms))

            """A function that plots the rooms on the floor plan"""
            # Create an image to draw the colored contours
            contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Get a colormap
            colormap = plt.get_cmap('prism', len(rooms))

            # Draw each contour with a different color and fill the inside
            for i, contour in enumerate(rooms):
                color = tuple(int(c * 255) for c in colormap(i)[:3])  # Convert colormap color to BGR
                cv2.drawContours(contour_image, [contour], -1, color, thickness=cv2.FILLED)

            # if image has 3 channels
            is_colored = len(image.shape) == 3

            # Convert the original image to RGB
            if is_colored:
                original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                original_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Create a mask where the contours are drawn
            mask = np.zeros_like(contour_image, dtype=np.uint8)
            for i, contour in enumerate(rooms):
                color = (255, 255, 255)  # White color for the mask
                cv2.drawContours(mask, [contour], -1, color, thickness=cv2.FILLED)

            # Blend the contour image with the original image using the mask
            alpha = 0.5  # Transparency factor
            plot_image = cv2.addWeighted(original_image_rgb, 1 - alpha, contour_image, alpha, 0)
        else:
            plot_image = image


        if save_file:
            cv2.imwrite(save_file, plot_image)
        else:
            # if image has 3 channels
            is_colored = len(plot_image.shape) == 3

            plt.figure(figsize=(15, 15))
            if is_colored:
                plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(plot_image, cmap='gray')
            plt.axis('off')
            plt.show()

    def calculate_contours(self):
        """
        A function that calculates the contours of the image
        """
        # Find the contours of the walls and doors
        contours, _ = cv2.findContours(self.walls_doors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.building = contours[0]
        self.rooms = contours[1:]

        # Reject small contours
        self.rooms = [room for room in self.rooms if cv2.contourArea(room) > 500]
        self.rooms = [room for room in self.rooms if cv2.contourArea(room) < 10000000]

        # Reject contours that circulate black areas
        def is_black_area(contour, image):
            # Create a mask with the contour
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            # Erode the mask
            mask = cv2.erode(mask, kernel(5), iterations=1)
            # A random black point of mask
            point = np.where(mask == 255)
            # Check if the point is black
            return image[point[0][0], point[1][0]] == 0

        self.rooms = [room for room in self.rooms if not is_black_area(room, self.walls_doors)]
        print('Number of rooms:', len(self.rooms))

        # Convert the contours to Shapely Polygons
        self.rooms = [Polygon(room.squeeze()) for room in self.rooms]

    def generate_graph(self, rooms=None, grid_size=30, radius=2.3):
        """
        A function that generates a graph based on the image
        """
        if rooms is not None:
            rooms_dict = {}
            for room in rooms:
                points = room['points']
                # Convert points to a list of tuples for Shapely Polygon
                polygon_points = [(point['x']*self.scale, point['y']*self.scale) for point in points]
                # Create a Shapely Polygon
                rooms_dict[room['label']] = Polygon(polygon_points)
        
            self.rooms = rooms_dict

        # Add a margin to the image
        margin = 10
        image_margin = cv2.copyMakeBorder(self.walls, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
        # Apply image dilation
        # image_margin = cv2.erode(image_margin, kernel(20), iterations=1)
        # Invert image in order not to avoid image border contour
        image_margin = cv2.bitwise_not(image_margin)
        contours, _ = cv2.findContours(image_margin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_margin = cv2.bitwise_not(image_margin)

        # Prepare the structuring element (kernel) only once
        ker = kernel(5)

        mask_margin = 5

        filtered_contours = []

        for contour in contours:
            # Get the bounding rectangle for this contour
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if the bounding box is empty (shouldn't happen in normal cases, but just a safeguard)
            if w == 0 or h == 0:
                continue

            x_new = max(x - mask_margin, 0)
            y_new = max(y - mask_margin, 0)
            w_new = min(w + mask_margin + x-x_new, image_margin.shape[1] - x)
            h_new = min(h + mask_margin + y-y_new, image_margin.shape[0] - y)
            
            # Extract the ROI from the original image
            sub_image_margin = image_margin[y_new:y_new+h_new, x_new:x_new+w_new]

            # Create a local mask the size of the bounding box
            mask_roi = np.full((h_new, w_new), 255, dtype=np.uint8)

            # Shift the contour so it fits exactly in the local mask
            contour_shifted = contour - [x_new, y_new]

            # Draw the contour on the local mask (fill with 0 = black)
            cv2.drawContours(mask_roi, [contour_shifted], -1, 0, thickness=cv2.FILLED)

            # Dilate the mask locally
            mask_roi = cv2.dilate(mask_roi, ker, iterations=1)

            # Combine the sub-image with the mask
            region = cv2.bitwise_or(sub_image_margin, mask_roi)

            # Check if there's any black pixel in the region
            # If so, we keep this contour
            if np.any(region == 0):
                filtered_contours.append(contour)

        contours = filtered_contours

        print('Contour Filtering Done')

        # Stage 6: Place the nodes on the image
        # Define the grid size
        self.grid_size = grid_size

        # Create nodes at the center of each grid cell
        nodes = []
        for i in range(grid_size, self.image_upscale.shape[1], grid_size):
            for j in range(grid_size, self.image_upscale.shape[0], grid_size):
                nodes.append((i, j))

        # Reject nodes that are on walls and doors
        valid_nodes = [(node[0]+margin, node[1]+margin) for node in nodes if self.walls[node[1], node[0]] == 255]

        # Stage 7: Create a graph based on the nodes
        # Connect the nodes to form a graph based on a threshold distance
        # Each node can be connected to its neighbors within a certain distance
        G = nx.Graph()
        for i, node in enumerate(valid_nodes):
            if self.rooms is not None:
                # Check if the node is inside a room
                node_point = Point(node)
                # TODO: Optimize this part, use STRtree. Also to this part at the end of the graph generation process (less nodes)
                for room, polygon in self.rooms.items():
                    if polygon.contains(node_point):
                        G.add_node(i, pos=node, label=room)
                        break
                else:
                    G.add_node(i, pos=node, label='Unknown')
            else:
                G.add_node(i, pos=node, label='Unknown')

        # Add nodes to the graph and create spatial index
        idx = index.Index()
        positions = {}
        for i, node in enumerate(valid_nodes):
            G.add_node(i)
            positions[i] = node
            idx.insert(i, (node[0], node[1], node[0], node[1]))

        # Convert contours to Shapely Polygons for efficient intersection checks
        # TODO: Reject extremely small contours @ this should get detected
        contours = [c for c in contours if cv2.contourArea(c) > 25]
        
        contours_poly = [Polygon(c.squeeze()) for c in contours]

        # Plot the contours_poly
        # if self._debug_:
        #     image = image_margin.copy()
        #     # Image to RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #     for contour in contours_poly:
        #         import random
        #         color = (random.randint(30, 200), random.randint(30, 200), random.randint(30, 200))
        #         cv2.drawContours(image, [np.array(contour.exterior.coords, dtype=np.int32)], -1, color, 2)
        #     self.plot_image(image, save_file='contours.png')

        # TODO: Keeping valid contours only
        # contours_poly = [c for c in contours_poly if c.is_valid]


        polygon_tree = STRtree(contours_poly)

        # Connect nodes if close enough and no intersection with contours
        for i, pos_i in positions.items():
            # Get the nearby nodes
            nearby_nodes = list(idx.nearest((pos_i[0], pos_i[1], pos_i[0], pos_i[1]), 50))  # Adjust the number based on needed proximity
            # Calculate distances once and store them in a dictionary
            distances_dict = {x: np.linalg.norm(np.array(pos_i) - np.array(positions[x])) for x in nearby_nodes}
            # Sort nearby_nodes based on the precomputed distances
            nearby_nodes = sorted(nearby_nodes, key=lambda x: distances_dict[x])
            # filter out nodes that are too far
            nearby_nodes = [x for x in nearby_nodes if distances_dict[x] <= grid_size * radius]
            for j in nearby_nodes:
                if i >= j:
                    continue
                pos_j = positions[j]
                vector = np.array(pos_j) - np.array(pos_i)
                vector //= grid_size
                # gcd of the vector components is 1, add the edge
                if np.gcd(vector[0], vector[1]) != 1:
                    continue

                line = LineString([pos_i, pos_j])
                # Efficient query to check if there are any intersecting polygons
                if not any(True for _ in polygon_tree.query(line, 'intersects')):
                    G.add_edge(i, j, weight=distances_dict[j])

        # # Plot connected nodes
        # if self._debug_:
        #     image = self.image_upscale.copy()
        #     # Image to RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #     for i, j in G.edges:
        #         cv2.line(image, (positions[i][0]-margin, positions[i][1]-margin), (positions[j][0]-margin, positions[j][1]-margin), (0, 0, 255), 2)
        #     for i, node in positions.items():
        #         cv2.circle(image, (node[0]-margin, node[1]-margin), 5, (255, 0, 0), -1)
        #     self.plot_image(image, save_file='graph.png')

        # Isolate the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        # # Get the new valid nodes
        graph_nodes = [positions[i] for i in G.nodes]

        # Create a mapping from the original node labels to new labels
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}

        G = nx.relabel_nodes(G, mapping)

        # # Plot connected nodes
        # if self._debug_:
        #     image = image_margin.copy()
        #     # Image to RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #     import numpy as np
        #     import random
        #     for contour in contours:
        #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #         cv2.drawContours(image, [contour], -1, color, 2)
        #     for i, j in G.edges:
        #         cv2.line(image, positions[i], positions[j], (0, 0, 255), 2)
        #     for i, node in positions.items():
        #         cv2.circle(image, node, 5, (255, 0, 0), -1)
        #     self.plot_image(image, save_file='graph.png')

        return G, graph_nodes, polygon_tree


    # TODO: implement this in order to avoid putting manually the scale factor
    def cm_to_pixels(self, cm, scale=1):
        # Convert centimeters to pixels
        return cm / self.pixel_to_cm * scale

    # TODO: implement this in order to avoid putting manually the scale factor
    def pixels_to_cm(self, pixels, scale=1):
        # Convert pixels to centimeters
        return pixels * self.pixel_to_cm / scale

## For testing purposes ##
if __name__ == '__main__':

    navigation = Indoor_Navigation('assets/images/hospital_1.jpg',
                                    'Fancy Hospital',
                                    debug=True)
    navigation.calibrate(0.00148)
    navigation.process_image(grid_size=30)

    # plot graph
    image = navigation.image_upscale.copy()
    graph = navigation.graph

    # image to rgb
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # draw edges
    for i, j in graph.edges:
        cv2.line(image, navigation.graph_nodes[i], navigation.graph_nodes[j], (255, 0, 0), 2)
    # draw nodes
    for node in navigation.graph_nodes:
        cv2.circle(image, node, 7, (0, 0, 255), -1)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.show()

    # navigation.plot_graph()
    #navigation.plot_graph()
    # print(navigation.report())
    # navigation.plot_rooms()
    # navigation = Indoor_Navigation('static/floor_plan_1.jpg',
    #                            'Demo floor plan',
    #                            grid_size=10)
    #navigation.calculate_and_plot_route((0, 0), (1, 1))
    # navigation.save('static/navigation.pkl')

    # navigation.save_json('navigation.json')



