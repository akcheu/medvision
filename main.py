from flask import Flask, render_template, request, session
from torchvision import transforms
from PIL import Image
import os
import torch
import clip
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = '123456789'
 
def model_predict(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    text = clip.tokenize(["a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    return probs

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/',  methods=("POST", "GET"))
def fetch_image():
    if request.method == 'POST':
        upload = request.files['uploaded-file']
        img_name = upload.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        upload.save(img_path)
        session['img_path'] = img_path
    return render_template('index2.html', user_image = img_path)

@app.route('/predict')
def predict():
    image = session.get('img_path', None)
    text = model_predict(image)
    return render_template('index3.html', user_image = image, user_text = text)

if __name__=='__main__':
    app.run(debug = True)