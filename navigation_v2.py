import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rtree import index
from shapely.geometry import Polygon, LineString
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
    name = None
    image_original = None
    image = None
    image_upscale = None
    walls = None
    doors = None
    walls_doors = None
    pixel_to_meters = None

    def __init__(self,
                 image_path: str,
                 name: str = None):
        """
        A class that represents an indoor navigation system
        """
        self.name = name
        self.image_original = cv2.imread(image_path)
    
    def process_image(self):
        """
        A function that processes the image
        """
        pbar = tqdm(total=3, desc="Overall Progress", unit="task")

        pbar.write("Initial image process...")
        self.image = clean_image(self.image_original)
        # TODO: Implement the upscale factor based on the pixel_to_meters ratio
        self.image_upscale = upscale_image(self.image)
        self.image_upscale = sharpen_image(self.image_upscale)
        pbar.update(1)
        pbar.write("Detecting walls...")
        walls = self.detect_walls(self.image_upscale, self.pixel_to_meters)
        pbar.update(1)
        pbar.write("Detecting doors...")
        self.doors, doors_mask = self.detect_doors()
        self.walls_doors = cv2.bitwise_and(walls, doors_mask)
        pbar.update(1)
        pbar.close()

    def calculate_route(self,
                        start: Tuple[float, float],
                        end: Tuple[float, float],
                        algorithm: Literal['dijkstra', 'astar'] = 'astar',
                        in_pixels: bool = False
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
            image_shape = self.image.shape
            start = (int(start[0] * image_shape[1]), int(start[1] * image_shape[0]))
            end = (int(end[0] * image_shape[1]), int(end[1] * image_shape[0]))

        # Find the nearest nodes to the start and end points
        start_node = min(self.graph_nodes, key=lambda x: np.sqrt((x[0] - start[0])**2 + (x[1] - start[1])**2))
        end_node = min(self.graph_nodes, key=lambda x: np.sqrt((x[0] - end[0])**2 + (x[1] - end[1])**2))

        # Find the shortest path
        if algorithm == 'dijkstra':
            path = nx.shortest_path(self.G, source=self.graph_nodes.index(start_node), target=self.graph_nodes.index(end_node), weight='weight')
        elif algorithm == 'astar':
            # Heuristic function for A* (Euclidean distance)
            def heuristic(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
                x1, y1 = self.G.nodes[node1]['pos']
                x2, y2 = self.G.nodes[node2]['pos']
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            # Find the shortest path using A* algorithm
            path = nx.astar_path(self.G, source=self.graph_nodes.index(start_node), target=self.graph_nodes.index(end_node), heuristic=heuristic, weight='weight')
        else:
            raise ValueError('Invalid algorithm. Choose either "dijkstra" or "astar"')
        
        # Calculate the total weight of the path
        distance = sum(self.G[u][v]['weight'] for u, v in zip(path, path[1:]))

        path_points = [self.graph_nodes[i] for i in path]

        # add starting and ending nodes to the path
        if in_pixels:
            start_point = start
            end_point = end
        else:
            start_point = (start[0] / image_shape[1], start[1] / image_shape[0])
            end_point = (end[0] / image_shape[1], end[1] / image_shape[0])

        # Check if start and end points can be connected directly to the path
        start_line = LineString([start_point, path_points[0]])
        end_line = LineString([end_point, path_points[-1]])
        if not does_line_intersect_contour(start_line, self.obstacles) and line_is_inside_contour(start_line, self.walls_contour):
            path_points.insert(0, start_point)
        if not does_line_intersect_contour(end_line, self.obstacles) and line_is_inside_contour(end_line, self.walls_contour):
            path_points.append(end_point)
        
        # Try skipping path points. Check if removing a point the path will pass through a contour
        i = 0
        j = len(path_points) - 1
        z = 0
        
        while i < j - 1:
            if z % 2 == 0:
                # Check from the start
                line = LineString([path_points[i], path_points[i + 2]])
                if not does_line_intersect_contour(line, self.obstacles) and line_is_inside_contour(line, self.walls_contour):
                    path_points.pop(i + 1)
                    j -= 1
                else:
                    i += 1
            else:
                # Check from the end
                line = LineString([path_points[j], path_points[j - 2]])
                if not does_line_intersect_contour(line, self.obstacles) and line_is_inside_contour(line, self.walls_contour):
                    path_points.pop(j - 1)
                    j -= 1
                else:
                    j -= 1
            
            z += 1
        
        i = 0
        while i < len(path_points) - 2:
            line = LineString([path_points[i], path_points[i + 2]])
            if not does_line_intersect_contour(line, self.obstacles) and line_is_inside_contour(line, self.walls_contour):
                path_points.pop(i+1)
            else:
                i += 1

        # fill pixels between points in the path to make it smooth
        # slice the distance between points till it is less than 10
        # i = 0
        # while i < len(path_points) - 1:
        #     if np.linalg.norm(np.array(path_points[i]) - np.array(path_points[i + 1])) > 50:
        #         inserted_point = ((path_points[i][0] + path_points[i + 1][0]) / 2, (path_points[i][1] + path_points[i + 1][1]) / 2)
        #         path_points.insert(i + 1, inserted_point)
        #     else:
        #         i += 1


        if not in_pixels:
            path_points = [(x / image_shape[1], y / image_shape[0]) for(x, y) in path_points]

        # Calculate the total distance of the path in meters by summing the the euclidean distances between the points
        distance = sum([np.linalg.norm(np.array(i) - np.array(j)) for i, j in zip(path_points, path_points[1:])])

        return path_points, distance


    def save(self, path: str) -> None:
        """A function that saves the object to a file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
   
    def save_json(self, path: str) -> None:
        """A function that saves the object to a JSON file"""
        rooms_dummy = ["Living Room", "Bathroom", "Kitchen", "Bedroom", "Office"]

        # TODO: Implement primary nodes, labels, rooms, floor
        data = { node : {"edges" : [], "label" : rooms_dummy[node % 5], "coords" : self.graph_nodes[node], "primary" : False} for node in self.G.nodes }

        for i, j in self.G.edges:
            data[i]["edges"].append(j)
            data[j]["edges"].append(i)

        json_graph = []

        # get items of dictionary
        for i, j in data.items():
            dict_i = {
                "id" : i,
                "edges" : j["edges"],
                "label" : j["label"],
                "imageX" : j["coords"][0] - self.margin,
                "imageY" : j["coords"][1] - self.margin,
                "primary" : j["primary"],
                "floor" : 0
            }
            json_graph.append(dict_i)
        
        # TODO: Implement room labels
        #json_rooms = [{"label": "Room", "coords": list(i.exterior.coords)} for i in self.rooms]
        json_rooms = [{"label": rooms_dummy[id], "floor": 0, "coords": [(int(x) - self.margin, int(y) - self.margin) for x, y in i.exterior.coords]} for id, i in enumerate(self.rooms)]

        json_data = {
            "Distance_Per_Pixel" : 1,
            "Graph" : json_graph,
            "Rooms" : json_rooms
        }

        with open(path, 'w') as f:
            json.dump(json_data, f, indent=4)


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
            f"-Number of graph edges: {len(self.G.edges)}"
        )

    def plot_image(self) -> None:
        """A function that plots the floor plan image"""
        plt.imshow(self.image, cmap='gray')
        plt.title('Floor Plan')
        plt.show()
    
    def plot_rooms(self) -> None:
        """A function that plots the rooms on the floor plan"""
        # Create an image to draw the colored contours
        contour_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

       # Get a colormap
        colormap = plt.get_cmap('rainbow', len(self.rooms))

        # Draw each contour with a different color and fill the inside
        for i, contour in enumerate(self.rooms):
            color = tuple(int(c * 255) for c in colormap(i)[:3])  # Convert colormap color to BGR
            cv2.drawContours(contour_image, [contour], -1, color, thickness=cv2.FILLED)

        # Convert the original image to RGB
        original_image_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        # Create a mask where the contours are drawn
        mask = np.zeros_like(contour_image, dtype=np.uint8)
        for i, contour in enumerate(self.rooms):
            color = (255, 255, 255)  # White color for the mask
            cv2.drawContours(mask, [contour], -1, color, thickness=cv2.FILLED)

        # Blend the contour image with the original image using the mask
        alpha = 0.5  # Transparency factor
        blended_image = cv2.addWeighted(original_image_rgb, 1 - alpha, contour_image, alpha, 0)

        # Plot the blended image
        plt.imshow(blended_image)
        plt.title('Room Detection')
        plt.axis('off')  # Hide axes for better visualization
        plt.show()

    def plot_graph(self) -> None:
        """A function that plots the graph on the floor plan"""
        # Draw the graph on the image
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos, node_color='red', with_labels=False, node_size=15)
        plt.imshow(self.image, cmap='gray')
        plt.title('Graph on Floor Plan')
        plt.show()
    
    def plot_route(self, path_points: List[Tuple[float, float]], distance: float, in_pixels: bool = False) -> None:
        """A function that plots the path on the floor plan"""
        if not in_pixels:
            image_shape = self.image.shape
            path_points = [(int(point[0] * image_shape[1]), int(point[1] * image_shape[0])) for point in path_points]
        plt.imshow(self.image, cmap='gray')
        for i in path_points:
            plt.plot(i[0], i[1], 'ro', markersize=1)
        plt.title('Shortest Path on Floor Plan\nDistance: {:.2f} meters'.format(distance))
        plt.show()
    
    def calculate_and_plot_route(self,
                                 start: Tuple[float, float],
                                 end: Tuple[float, float],
                                 algorithm: Literal['dijkstra', 'astar'] = 'astar',
                                 in_pixels: bool = False) -> None:
        """A function that calculates and plots the shortest path between two points"""
        path_points, distance = self.calculate_route(start, end, algorithm, in_pixels)
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

            predictions = CLIENT.infer(image_path, model_id="full-set-menu/5")['predictions']
            # predictions = CLIENT.infer(image_path, model_id="doors-windows-detection/4")['predictions']

        # Filter the predictions to get the classes
        classes = [p for p in predictions if p['class'] in classes and p['confidence'] > confidence]

        return classes

    def detect_doors(self):
        """
        A function that returns the doors detected in the image
        """
        # doors = self.predict_class(['door'])
        doors_all = self.predict_class(['DOOR-SINGLE', 'DOOR-DOUBLE', 'WINDOW'])
        doors = [d for d in doors_all if d['class'] == 'DOOR-SINGLE']
        doors2 = [d for d in doors_all if d['class'] == 'DOOR-DOUBLE']
        windows = [d for d in doors_all if d['class'] == 'WINDOW']
        mask = np.zeros((self.image_upscale.shape[0], self.image_upscale.shape[1]), dtype=np.uint8)
        centers = []

        factor = 1.4
        for i, door in enumerate(doors):
            x, y, width, height = door['x'], door['y'], door['width'], door['height']

            x1 = int(x - width * factor / 2)
            y1 = int(y - height * factor / 2)
            x2 = int(x + width * factor / 2)
            y2 = int(y + height * factor / 2)

            image = self.image_upscale[y1:y2, x1:x2]

            # Add margin to the image
            margin = max(x2-x1, y2-y1)

            image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)

            door = apply_filters(image, [in_range(0, 80)])
            door_open = apply_filters(door, [morph_dilate(kernel(80)), morph_erode(kernel(60))])
            door_xor = cv2.bitwise_xor(door, door_open)
            door_dilated = apply_filters(door, [morph_dilate(kernel(30)), invert])
            door_final = cv2.bitwise_and(door_open, door_dilated)
            contour = cv2.findContours(door_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                # keep the contour with the largest area
                contour = max(contour[0], key=cv2.contourArea)

                # find the center of the contour
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # draw center on the image
                img = image.copy()
                img[cy, cx] = 0

                door2 = apply_filters(img, [in_range(0, 80)])
                door_open2 = apply_filters(door2, [morph_dilate(kernel(50)), morph_erode(kernel(40))])
                door_xor2 = cv2.bitwise_xor(door2, door_open2)
                door_dilated2 = apply_filters(door, [invert])

                door_final2 = cv2.bitwise_and(door_open2, door_dilated2)

                mask[y1:y2, x1:x2] = door_open2[margin:-margin, margin:-margin]

                centers.append((cx, cy))

            except:
                pass

        for i, door in enumerate(windows):
            x, y, width, height = door['x'], door['y'], door['width'], door['height']

            x1 = int(x - width * factor / 2)
            y1 = int(y - height * factor / 2)
            x2 = int(x + width * factor / 2)
            y2 = int(y + height * factor / 2)

            image = self.image_upscale[y1:y2, x1:x2]

            # Add margin to the image
            margin = max(x2-x1, y2-y1)

            image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)

            door = apply_filters(image, [in_range(0, 80)])
            door_open = apply_filters(door, [morph_dilate(kernel(80)), morph_erode(kernel(60))])
            door_xor = cv2.bitwise_xor(door, door_open)
            door_dilated = apply_filters(door, [morph_dilate(kernel(30)), invert])
            door_final = cv2.bitwise_and(door_open, door_dilated)
            contour = cv2.findContours(door_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                # keep the contour with the largest area
                contour = max(contour[0], key=cv2.contourArea)

                # find the center of the contour
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # draw center on the image
                img = image.copy()
                img[cy, cx] = 0

                door2 = apply_filters(img, [in_range(0, 80)])
                door_open2 = apply_filters(door2, [morph_dilate(kernel(50)), morph_erode(kernel(40))])
                door_xor2 = cv2.bitwise_xor(door2, door_open2)
                door_dilated2 = apply_filters(door, [invert])

                door_final2 = cv2.bitwise_and(door_open2, door_dilated2)

                mask[y1:y2, x1:x2] = door_open2[margin:-margin, margin:-margin]

            except:
                pass

        
        for i, door in enumerate(doors2):
            x, y, width, height = door['x'], door['y'], door['width'], door['height']

            x1 = int(x - width * factor / 2)
            y1 = int(y - height * factor / 2)
            x2 = int(x + width * factor / 2)
            y2 = int(y + height * factor / 2)

            image = self.image_upscale[y1:y2, x1:x2]

            # Add margin to the image
            margin = max(x2-x1, y2-y1)

            image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)

            door = apply_filters(image, [in_range(0, 80)])
            door_open = apply_filters(door, [morph_dilate(kernel(120)), morph_erode(kernel(100))])
            door_xor = cv2.bitwise_xor(door, door_open)
            door_dilated = apply_filters(door, [morph_dilate(kernel(30)), invert])
            door_final = cv2.bitwise_and(door_open, door_dilated)
            contour = cv2.findContours(door_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                # keep the contour with the largest area
                contour = max(contour[0], key=cv2.contourArea)

                # find the center of the contour
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # draw center on the image
                img = image.copy()
                img[cy, cx] = 0

                door2 = apply_filters(img, [in_range(0, 80)])
                door_open2 = apply_filters(door2, [morph_dilate(kernel(80)), morph_erode(kernel(70))])
                door_xor2 = cv2.bitwise_xor(door2, door_open2)
                door_dilated2 = apply_filters(door, [invert])

                door_final2 = cv2.bitwise_and(door_open2, door_dilated2)

                mask[y1:y2, x1:x2] = door_open2[margin:-margin, margin:-margin]

                centers.append((cx, cy))

            except:
                pass

        mask = cv2.bitwise_not(mask)

        return centers, mask

    # TODO: Implement the pixel_to_meters
    def detect_walls(self, image, pixel_to_meters):
        # detect walls
        filters = [
                threshold(100, 255),
                ]
        walls_1 = apply_filters(image, filters)

        # detect walls
        filters = [
                in_range(120, 190),
                invert,
                gaussian_blur((21, 21), 6),
                threshold(180, 255),
                morph_open(kernel(41)),
                ]
        walls_2 = apply_filters(image, filters)

        # mask OR image
        walls = cv2.bitwise_and(walls_1, walls_2)

        filters = [
                gaussian_blur((13, 13), 6),
                threshold(200, 255),
                ]
        walls = apply_filters(walls, filters)

        # Remove external objects
        filters = [
                morph_open(kernel(171)),
                invert
                ]
        mask = apply_filters(walls, filters)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # keep the contour with the largest area
        building = max(contours, key=cv2.contourArea)

        # create a mask with the building
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [building], -1, 255, -1)

        filters = [
            invert
        ]
        mask = apply_filters(mask, filters)

        walls = cv2.bitwise_or(walls, mask)

        return walls



## For testing purposes ##
if __name__ == '__main__':
    navigation = Indoor_Navigation('assets/images/hospital_1.jpg',
                                   'Fancy Hospital')
    navigation.process_image()
    # save image with doors
    # cv2.imwrite('assets/images/hospital_1_walls_doors.jpg', navigation.walls_doors)
    plt.figure()
    plt.imshow(navigation.walls_doors, cmap='gray')
    plt.axis('off')
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



