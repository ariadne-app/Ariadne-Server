from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import json

app = Flask(__name__)

# Path to image dataset
<<<<<<< HEAD
#IMAGE_FOLDER = "../YOLO/Floor-Plan/test/images"
IMAGE_FOLDER = "assets/images"
# Path to save bounding box labels
LABELS_FOLDER = "assets/labels"
# LABELS_FOLDER = "../YOLO/Floor-Plan/test/labels"
=======
# IMAGE_FOLDER = "../YOLO/Floor-Plan/train/images"
IMAGE_FOLDER = "assets/images"
# Path to save bounding box labels
LABELS_FOLDER = "assets/labels"
# LABELS_FOLDER = "../YOLO/Floor-Plan/train/labels"
>>>>>>> 9df3288 (Add files)

# Load image file names
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('jpg', 'jpeg', 'png'))])
current_image_index = 0  # Tracks the current image index

if not os.path.exists(LABELS_FOLDER):
    os.makedirs(LABELS_FOLDER)

for image in image_files:
<<<<<<< HEAD
    label_file = os.path.join(LABELS_FOLDER, f"{image.split('.')[0]}.txt")
    if not os.path.exists(label_file):
        with open(label_file, 'w') as f:
=======
    file_id = image.split('.')[0]
    folder_files = os.listdir(LABELS_FOLDER)
    # check if there is a file starting with the image name
    if not any([file_id in file for file in folder_files]):
        with open(os.path.join(LABELS_FOLDER, file_id+'.txt'), 'w') as f:
>>>>>>> 9df3288 (Add files)
            f.write("")

boxes_files = sorted([f for f in os.listdir(LABELS_FOLDER) if f.endswith('txt')])

# .txt files contain bounding box coordinates in the format:
# <class_id> <righttopx> <righttopy> <leftbottomx> <leftbottomy>
boxes = []
for file in boxes_files:
    with open(os.path.join(LABELS_FOLDER, file), 'r') as f:
        text = f.readlines()
        box_list = []
        for line in text:
            # Split the line once and unpack the values
            values = line.split()
            class_id = int(values[0])
            # Convert the rest of the values to float and create the bounding box list
            box = [class_id] + list(map(float, values[1:]))
            box_list.append(box)
        boxes.append(box_list)


@app.route('/')
def index():
    global current_image_index
    return render_template('labeling.html', image=image_files[current_image_index])

# Render any static file
@app.route('/assets/<path:path>')
def static_file(path):
    return send_from_directory('assets', path)

# Render any static file
@app.route('/YOLO/<path:path>')
def yolo_static_file(path):
    return send_from_directory('../YOLO', path)

@app.route('/first_image')
def first_image():
    image_index = 0

    # Manually construct the URL for the image
    image_url = f"assets/../{IMAGE_FOLDER}/{image_files[image_index]}"

    response_data = {
        'image_index': image_index,
        'image_url': image_url,
        'boxes': boxes[image_index]
    }
    return jsonify(response_data)

@app.route('/next_image')
def next_image():
    # get image index from the get request
    image_index = int(request.args.get('image_index'))

    if image_index < len(image_files) - 1:
        image_index += 1
    else:
        image_index = 0

    # Manually construct the URL for the image
    image_url = f"assets/../{IMAGE_FOLDER}/{image_files[image_index]}"

    response_data = {
        'image_index': image_index,
        'image_url': image_url,
        'boxes': boxes[image_index]
    }
    return jsonify(response_data)

@app.route('/prev_image')
def prev_image():
    # get image index from the get request
    image_index = int(request.args.get('image_index'))

    if image_index > 0:
        image_index -= 1
    else:
        image_index = len(image_files) - 1

    # Manually construct the URL for the image
    image_url = f"assets/../{IMAGE_FOLDER}/{image_files[image_index]}"

    response_data = {
        'image_index': image_index,
        'image_url': image_url,
        'boxes': boxes[image_index]
    }
    return jsonify(response_data)

@app.route('/save_box', methods=['POST'])
def save_box():
    data = request.json
    bbox = data.get('box')

    box = [int(bbox['object_id']), float(bbox['centerx']), float(bbox['centery']), float(bbox['width']), float(bbox['height'])]

    image_index = int(data.get('image_index'))

    # Append the new box to the list of boxes
    boxes[image_index].append(box)
    
    box_file = os.path.join(LABELS_FOLDER, f"{image_files[image_index].split('.')[0]}.txt")

    # Append the new box to the file
    with open(box_file, 'a') as f:
        f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")
    
    return jsonify({"status": "success"})

@app.route('/delete_box', methods=['POST'])
def delete_box():
    data = request.json
    bbox = data.get('box')
    image_index = int(data.get('image_index'))

    box = [int(bbox['object_id']), float(bbox['centerx']), float(bbox['centery']), float(bbox['width']), float(bbox['height'])]

    def compare_boxes(box1, box2):
        def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        return box1[0] == box2[0] and is_close(box1[1], box2[1]) and is_close(box1[2], box2[2]) and is_close(box1[3], box2[3]) and is_close(box1[4], box2[4])

    # Search for the box in the list of boxes and remove it
    removed = False
    for i, b in enumerate(boxes[image_index]):
        if compare_boxes(b, box):
            boxes[image_index].pop(i)
            removed = True
            break

    if not removed:
        return jsonify({"status": "error"}, 400)

    box_file = os.path.join(LABELS_FOLDER, f"{image_files[image_index].split('.')[0]}.txt")

    # Write the remaining boxes to the file
    with open(box_file, 'w') as f:
        for box in boxes[image_index]:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True)
