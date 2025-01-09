from flask import Flask, request, jsonify
import threading
import time
import requests
# from super_gradients.training import models
import cv2
import numpy as np
import os
import base64
import subprocess
import logging
import shutil
from predict_walls.UNet_Pytorch_Customdataset.predict import predict_wall_mask
from scripts.navigation import Indoor_Navigation

from ultralytics import YOLO

from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://ariadne-admin.web.app"])

BASE_PATH = os.path.dirname(__file__)  # Get the directory of the current file


def process_predictions(model_result):

    label_names = {
        2: "door",
    }

    bboxes = model_result.prediction.bboxes_xyxy.tolist()
    confidence = model_result.prediction.confidence.tolist()
    labels = model_result.prediction.labels.tolist()

    filtered_bboxes = [bbox for bbox, label in zip(bboxes, labels) if label == 2]
    filtered_confidence = [conf for conf, label in zip(confidence, labels) if label == 2]
    filtered_labels = [label for label in labels if label == 2]
    
    formatted_bboxes = [
        {
            "x": bbox[0],  
            "y": bbox[1],  
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
            "label": label_names.get(label, "unknown")
        }
        for bbox, label in zip(filtered_bboxes, filtered_labels)
    ]

    response = {
        "bboxes": filtered_bboxes,
        "confidence": filtered_confidence,
        "labels": filtered_labels,
        "formatted_bboxes": formatted_bboxes
    }
    
    return jsonify(response)

def process_predictions_yolo11(boxes):
    #boxes[0].xyxy.tolist()[0]
    formatted_bboxes = [
        {
            "x": box.xyxy.tolist()[0][0],
            "y": box.xyxy.tolist()[0][1],
            "width": box.xyxy.tolist()[0][2] - box.xyxy.tolist()[0][0],
            "height": box.xyxy.tolist()[0][3] - box.xyxy.tolist()[0][1],
            "label": 'door'
        }
        for box in boxes
    ]

    response = {
        "formatted_bboxes": formatted_bboxes
    }
    
    return jsonify(response)

def convert_txt_to_json(txt_file_path, image_width, image_height):
    data = []
    app.logger.info(f"Image dimensions: width={image_width}, height={image_height}")
    with open(txt_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1:
                class_id = int(parts[0])
                points = list(map(float, parts[1:]))
                point_objects = []
                for i in range(0, len(points), 2):
                    x = points[i] * image_width
                    y = points[i + 1] * image_height
                    point_objects.append({"x": x, "y": y})
                data.append({"points": point_objects, "label": "Uknown"})
    return jsonify(data)

# Test endpoint
@app.route('/test_endpoint', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Test successful!"})


def make_request_periodically():
    while True:
        try:
            response = requests.get('http://127.0.0.1:{}/test_endpoint'.format(int(os.environ.get("PORT", 5000))))  # Assuming your Flask app runs locally on port 5000
            if response.status_code == 200:
                app.logger.info("Successfully made request to /test_endpoint")
            else:
                app.logger.error(f"Failed to make request to /test_endpoint. Status code: {response.status_code}")
        except Exception as e:
            app.logger.error(f"Error making request to /test_endpoint: {str(e)}")
        
        time.sleep(300)  # Sleep for 5 minutes (300 seconds)


@app.route('/predict_rooms', methods=['POST'])
def predict_rooms():
    try:
        model_path = os.path.join(BASE_PATH, "models", "best.pt")
        
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        if 'imageDimensions' not in data:
            return jsonify({"error": "No image dimensions data provided"}), 400
        
        base64_string = data['image']
        image_dimensions = data['imageDimensions']

        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)

        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        height, width, _ = image.shape

        temp_path = os.path.join(BASE_PATH, "assets", "temp_images")
        temp_image_path = os.path.join(temp_path, 'temp_image_rooms.jpg')
    
        success = cv2.imwrite(temp_image_path, image)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        yolo_command = [
            "yolo",
            "task=segment",
            "mode=predict",
            f"model={model_path}",
            "conf=0.5",
            f"source={os.path.join(temp_path, 'temp_image_rooms.jpg')}",
            "save=True",
            "save_txt=True",  
            f"project={temp_path}",  
            "show_labels=False",
            "show_conf=False",
            "show_boxes=False"
        ]
        predict_path = os.path.join(temp_path, "predict")
        try:
            shutil.rmtree(predict_path)
            app.logger.info(f"Folder '{predict_path}' deleted successfully.")
        except FileNotFoundError:
            app.logger.info(f"Folder '{predict_path}' already deleted.")
        
        result = subprocess.run(yolo_command, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            return jsonify({"error": "YOLO command failed", "details": result.stderr}), 500
        
        txt_path = os.path.join(predict_path, 'labels')
        
        txt_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
        if not txt_files:
            return jsonify({"error": "No .txt file generated"}), 500
        txt_file_path = os.path.join(txt_path, txt_files[0])

        response = convert_txt_to_json(txt_file_path, image_dimensions['width'], image_dimensions['height'])

        shutil.rmtree(predict_path)

        return response

    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route('/predict_doors_yolo11', methods=['POST'])
def predict_doors_yolo11():

    model_path = os.path.join(BASE_PATH, "models", "full_set_menu-yolo11m_plus2.pt")

    data = request.get_json()

    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    base64_string = data['image']
    
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)

    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    model = YOLO(model_path)

    results = model(image)[0]

    doors_classes = [1, 2]
    boxes = [box for box in results.boxes if int(box.cls) in doors_classes]

    return process_predictions_yolo11(boxes)

@app.route('/post_user_feedback', methods=['POST'])
def post_user_feedback():
    data = request.get_json()
    doors = data['doors']
    rooms = data['rooms']
    pixels_to_cm = float(data['distancePerPixel'])*100.0

    image_path = os.path.join(BASE_PATH, "assets", "temp_images", "temp_image_rooms.jpg") 

    navigation = Indoor_Navigation(image_path)
    navigation.calibrate(pixels_to_cm)
    grid_size = int(navigation.cm_to_pixels(150, scale=4))
    navigation.process_image(grid_size=grid_size, doors=doors, rooms=rooms)

    navigation.save('navigation-instances/admin_ui_test.pkl')
    
    return jsonify({"graph" : navigation.get_json()})



if __name__ == '__main__':
    threading.Thread(target=make_request_periodically, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
