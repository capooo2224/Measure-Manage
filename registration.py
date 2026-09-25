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
    img_type = flask.request.form.get('ImageType', '').strip()
    image = flask.request.files.get('Image')

    if not all((last_name, first_name, student_number, section, img_type, image)):
        return 'All fields are required.', 400

    if not image.filename or not allowed_file(image.filename):
        return "Invalid image file type!", 400

    image_data = image.read()
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        return 'The uploaded file is not a valid image.', 400

    student_name = f"{first_name}{last_name}"
    filename = f"{student_number}_{student_name}_{img_type}.jpg"
    
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


@app.route('/export_csv', methods=['GET'])
def export_csv():
    folder_name = flask.request.args.get('folder_name', '').strip()
    folder_path = os.path.join(EXPECTED_FACES_DIR, folder_name)

    if not os.path.exists(folder_path):
        return "Section folder not found.", 404

    # Check for existing CSV in folder or generate one from registered images
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if csv_files:
        # Return the latest recorded CSV file
        latest_csv = sorted(csv_files)[-1]
        return flask.send_from_directory(folder_path, latest_csv, as_attachment=True)

    # Generate a temporary student list CSV if no log exists
    csv_path = os.path.join(folder_path, f"{folder_name}_roster.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Student ID', 'Name', 'Image Type', 'Filename'])
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = fname.rsplit('.', 1)[0].split('_')
                sid = parts[0] if len(parts) > 0 else 'N/A'
                name = parts[1] if len(parts) > 1 else 'N/A'
                itype = parts[2] if len(parts) > 2 else 'N/A'
                writer.writerow([sid, name, itype, fname])

    return flask.send_from_directory(folder_path, f"{folder_name}_roster.csv", as_attachment=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    app.run(debug=True)