import os
from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
from pathlib import Path

# Create a directory if doesnt exists
if not os.path.exists("./uploads/"):
    os.makedirs("./uploads/")

# Create a flask app   
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Save the file
            file_path = Path("./uploads/") / uploaded_file.filename
            uploaded_file.save(file_path)

            # Load and preprocess the uploaded image
            img = image.load_img(file_path, target_size=(224, 224))
            image_array = image.img_to_array(img)
            train = np.expand_dims(image_array, axis=0)
            train = vgg16.preprocess_input(train)

            # Load the VGG16 model
            model = vgg16.VGG16(weights="imagenet")

            # Make predictions
            prediction = model.predict(train)
            pred = vgg16.decode_predictions(prediction)

            # Render the results
            return render_template('result.html', image_path=file_path, predictions=pred)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
