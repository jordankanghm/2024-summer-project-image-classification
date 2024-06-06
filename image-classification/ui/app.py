from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
import sys

sys.path.append('../utils')
from models import SimpleCNN

app = Flask(__name__)

# Load the best model weights
model = SimpleCNN()
model_path = os.path.join('../utils', 'best_simple_model_weights.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
TEST_IMAGES_DIR = 'static/test-images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    # Load the image and convert to grayscale
    img = Image.open(filepath).convert('L')

    # Transform the images
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = transform(img)

    # Add a batch dimension (1, 1, 28, 28)
    img = img.unsqueeze(0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = preprocess_image(filepath)

            with torch.no_grad():
                output = model(img)
            predicted_class = torch.argmax(output, dim=1).item()

            return render_template('result.html', filename=filename, predicted_class=predicted_class)

    return render_template('index.html')

@app.route('/select_test_image')
def select_test_image():
    test_images = os.listdir(os.path.join(app.static_folder, 'test-images'))
    test_images = [img for img in test_images if allowed_file(img)]

    return render_template('select_test_image.html', images=test_images)

@app.route('/predict_test_image/<filename>')
def predict_test_image(filename):
    filepath = os.path.join(app.static_folder, 'test-images', filename)
    img = preprocess_image(filepath)

    with torch.no_grad():
        output = model(img)
    predicted_class = torch.argmax(output, dim=1).item()

    return render_template('result.html', filename=filename, predicted_class=predicted_class, test_image=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), filename)

@app.route('/static/test-images/<path:filename>')
def serve_test_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'test-images'), filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
