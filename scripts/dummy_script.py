import base64
import requests

def send_image_to_endpoint(image_path, endpoint_url):
    try:
        # Convert the image to Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Create the payload
        payload = {
            "image": base64_image
        }

        # Make the POST request
        response = requests.post(endpoint_url, json=payload)

        # Check the response
        if response.status_code == 200:
            print("Request successful!")
            print("Response:", response.json())
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response:", response.text)

    except Exception as e:
        print("An error occurred:", str(e))

# Example usage
if __name__ == "__main__":
    IMAGE_PATH = "C:/Users/thano/Indoor-Navigation/software/web interface/assets/images/floor_plan_1.jpg"  # Replace with the path to your image
    ENDPOINT_URL = "http://127.0.0.1:5000/predict_doors"  # Replace with your endpoint URL
    send_image_to_endpoint(IMAGE_PATH, ENDPOINT_URL)
