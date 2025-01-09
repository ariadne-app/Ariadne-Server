from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from PIL import Image, ImageDraw
import os
from navigation import Indoor_Navigation
import numpy as np


app = Flask(__name__)


# Route to render the initial page
@app.route('/')
def home():
    return render_template('index.html')

# Render any static file
@app.route('/assets/<path:path>')
def static_file(path):
    return send_from_directory('assets', path)


# Route to handle the coordinate submission and return the path points
@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    data = request.json
    point_1 = (data['point1']['x'], data['point1']['y'])
    point_2 = (data['point2']['x'], data['point2']['y'])
    
    path_points, distance = navigation.calculate_route(point_1, point_2, in_pixels=True)

    # keep two decimal points
    distance = round(distance*0.027, 2)

    return jsonify({
        'path_points': path_points,
        'distance': distance
    })

# Route to reload the navigation object
@app.route('/reload_navigation', methods=['POST'])
def reload_navigation():
    global navigation
    navigation = Indoor_Navigation.load('navigation-instances/navigation.pkl')
    return jsonify({})


if __name__ == '__main__':
    navigation = Indoor_Navigation.load('navigation-instances/navigation.pkl')
    app.run(debug=True)
