import os
import torch
import numpy as np
import cv2
from flask import Flask, request, render_template

app = Flask(__name__)

model = torch.load("model.pt")
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join("uploads", file.filename))

        image = cv2.imread(os.path.join("uploads", file.filename))
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        
        with torch.no_grad():
            prediction = model(image)
        
        return str(prediction[0].numpy())

if __name__ == "__main__":
    app.run(debug=True)