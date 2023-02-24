from flask import Flask, render_template, request, session
from torchvision import transforms
from PIL import Image
import os
import torch
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = '123456789'
 
def model_predict(image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()
    input_image = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    text = str(categories[top1_catid[0]]) + " " + str(top1_prob[0].item())
    return text

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