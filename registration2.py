import os
import csv
import cv2
import flask
import numpy as np

app = flask.Flask(__name__)

EXPECTED_FACES_DIR = os.path.join(app.root_path, 'Expected_faces_db')
os.makedirs(EXPECTED_FACES_DIR, exist_ok=True)

def get_subfolders():
    """Dynamically fetch all folder names inside Expected_faces_db."""
    return [
        d for d in os.listdir(EXPECTED_FACES_DIR)
        if os.path.isdir(os.path.join(EXPECTED_FACES_DIR, d)) and not d.startswith('.')
    ]

@app.route('/', methods=['GET'])
def registration_form():
    folders = get_subfolders()
    return flask.render_template('registration.html', folders=folders)


@app.route('/register', methods=['POST'])
def register():
    last_name = flask.request.form.get('LastName', '').strip()
    first_name = flask.request.form.get('FirstName', '').strip()
    student_number = flask.request.form.get('StudentNumber', '').strip()
    section = flask.request.form.get('Section', '').strip()
    image = flask.request.files.get('Image')

    if not all((last_name, first_name, student_number, section, image)):
        return 'All fields are required.', 400

    if not image.filename or not allowed_file(image.filename):
        return "Invalid image file type!", 400

    image_data = image.read()
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        return 'The uploaded file is not a valid image.', 400

    student_name = f"{last_name}, {first_name}"
    filename = f"{student_number}_{student_name}.jpg"
    
    section_dir = os.path.join(EXPECTED_FACES_DIR, section)
    os.makedirs(section_dir, exist_ok=True)
    output_path = os.path.join(section_dir, filename)

    if not cv2.imwrite(output_path, image_array):
        return 'Could not save image.', 500

    return flask.redirect(flask.url_for('registration_form'))

@app.route('/add_folder', methods=['POST'])
def add_folder():
    folder_name = flask.request.form.get('folder_name', '').strip()
    if folder_name:
        folder_path = os.path.join(EXPECTED_FACES_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    return flask.redirect(flask.url_for('registration_form'))


@app.route('/rename_folder', methods=['POST'])
def rename_folder():
    old_name = flask.request.form.get('old_folder_name', '').strip()
    new_name = flask.request.form.get('new_folder_name', '').strip()

    if old_name and new_name:
        old_path = os.path.join(EXPECTED_FACES_DIR, old_name)
        new_path = os.path.join(EXPECTED_FACES_DIR, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)

    return flask.redirect(flask.url_for('registration_form'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)