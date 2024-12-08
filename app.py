from flask import Flask, render_template, request, url_for
import os
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Firebase Initialization
cred = credentials.Certificate('models/capstone-92833-firebase-adminsdk-s3jx2-ecca31e283.json')  # Path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://capstone-92833-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Define the CNN model model.pth
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 64 filters
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 128 filters
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Adjust dimensions based on input size (e.g., 128x128 image input)
        self._to_linear = 128 * (128 // 8) * (128 // 8)
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Output layer with 5 classes

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self._to_linear)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# # Define the CNN model model.pth

# Load the model and fix checkpoint issues
model_path = "models/cnn_model123.pth"
class_names = ['Dolichocephalocyrtus', 'Metapocyrtus', 'Orthocyrtus', 'Pachyrhynchus', 'Trachycyrtus']
num_classes = len(class_names)

model = CNN(num_classes=num_classes).to(device)

# Load checkpoint and fix if there are extra class weights
checkpoint = torch.load(model_path, map_location=device)
if checkpoint['fc2.weight'].size(0) != num_classes:
    logging.warning(f"Adjusting checkpoint to match {num_classes} classes.")
    checkpoint['fc2.weight'] = checkpoint['fc2.weight'][:num_classes, :]
    checkpoint['fc2.bias'] = checkpoint['fc2.bias'][:num_classes]

model.load_state_dict(checkpoint)
model.eval()
logging.info("Model loaded successfully.")

# Define image directory
images_dir = "./static/images"
os.makedirs(images_dir, exist_ok=True)

# Define preprocessing for the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for genusdetails page
@app.route('/genusdetails/<key>')
def genusdetails(key):
    # Fetch data from 'species/genera' in Firebase
    species_ref = db.reference('species/genera')
    genus_data = species_ref.get()

    species_details = None
    genus_name = None

    # Search through all genera to find the speciesKey
    for genus, species in genus_data.items():
        if key in species:
            species_details = species[key]
            genus_name = genus
            break

    if species_details:
        return render_template('details/genusdetails.html', species_data=species_details, genus_name=genus_name)
    else:
        return 'Species or Genus not found', 404


    
    
# Route for subgenusdetails page
@app.route('/subgenusdetails/<key>')
def subgenusdetails(key):
    # Fetch data from 'species/subgenera' in Firebase
    species_ref = db.reference('species/subgenera')
    subgenus_data = species_ref.get()

    species_details = None
    subgenus_name = None

    # Search through all subgenera to find the speciesKey
    for subgenus, species in subgenus_data.items():
        if key in species:
            species_details = species[key]
            subgenus_name = subgenus
            break

    if species_details:
        return render_template('details/subgenusdetails.html', species_data=species_details, subgenus_name=subgenus_name)
    else:
        return 'Species or Subgenus not found', 404




@app.route('/genera')
def genera():
    return render_template('weevils/genera.html')



@app.route('/sub_genus')
def sub_genus():
    return render_template('weevils/sub-genus.html')

@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html', prediction=None, confidence=None, image_path=None, note=None)

# Route for prediction
@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Validate if the uploaded file is an image
        imagefile = request.files.get('imagefile')
        if not imagefile or not imagefile.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            note = "Invalid image format. Please upload a PNG or JPEG image."
            logging.warning(note)
            return render_template('prediction.html', prediction=None, confidence=None, image_path=None, note=note)

        # Save the image
        image_path = os.path.join(images_dir, imagefile.filename)
        with open(image_path, 'wb') as f:
            f.write(imagefile.read())
        logging.info(f"Image saved: {image_path}")

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = probabilities.argmax()
            confidence = probabilities[predicted_idx] * 100

        # Get predicted genus and confidence
        predicted_genus = class_names[predicted_idx]
        confidence_display = round(confidence, 2)
        note = "" if confidence >= 50 else "Low confidence in prediction."

        # Generate the URL for the image
        image_url = url_for('static', filename='images/' + imagefile.filename)

        return render_template('prediction.html', prediction=predicted_genus, confidence=confidence_display, image_path=image_url, note=note)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return render_template('prediction.html', prediction=None, confidence=None, image_path=None, note="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
