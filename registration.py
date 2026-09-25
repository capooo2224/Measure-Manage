import os
import cv2
import flask
import numpy as np


app = flask.Flask(__name__)
section_a = os.path.join(app.root_path, 'Section_A')
section_b = os.path.join(app.root_path, 'Section_B')
section_c = os.path.join(app.root_path, 'Section_C')
section_directories = {
    'SectionA': section_a,
    'SectionB': section_b,
    'SectionC': section_c,
}


@app.route('/', methods=['GET'])
def registration_form():
    return flask.send_from_directory(app.root_path, 'registration.html')

@app.route('/register', methods=['POST'])
def register():
    # Get the form data
    last_name = flask.request.form.get('LastName', '').strip()
    first_name = flask.request.form.get('FirstName', '').strip()
    student_number = flask.request.form.get('StudentNumber', '').strip()
    section = flask.request.form.get('Section', '').strip()
    image = flask.request.files.get('Image')

    if not all((last_name, first_name, student_number, section, image)):
        return 'All fields are required.', 400

    # Check if the image file is allowed
    if not image.filename or not allowed_file(image.filename):
        return "Invalid image file type!", 400

    # Process the image
    image_data = image.read()
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        return 'The uploaded file is not a valid image.', 400

    processed_image = image_processing(image_array)

    # Use all registration details to identify the saved image.
    filename = f"{student_number}_{last_name}_{first_name}.jpg"
    save_image(processed_image, filename, section)

    return flask.redirect(flask.url_for('registration_form'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def image_processing(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges

def save_image(image, filename, section):
    section_directory = section_directories.get(section)
    if section_directory is None:
        raise ValueError("Invalid section specified. Must be 'SectionA', 'SectionB', or 'SectionC'.")

    os.makedirs(section_directory, exist_ok=True)
    output_path = os.path.join(section_directory, filename)
    base_name, extension = os.path.splitext(filename)
    counter = 2
    while os.path.exists(output_path):
        output_path = os.path.join(section_directory, f'{base_name}_{counter}{extension}')
        counter += 1

    if not cv2.imwrite(output_path, image):
        raise OSError(f'Could not save processed image to {output_path}')

if __name__ == '__main__':
    app.run(debug=True)

