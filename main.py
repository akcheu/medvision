from flask import Flask, render_template, request, session
from torchvision import transforms
from PIL import Image
from docarray import DocumentArray, Document

import os
import torch
import clip
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = '123456789'
 
def model_predict(image):
    embed_loc = 'finetune-mimic-clip-0/'
    train_text_da = DocumentArray.load(embed_loc + 'train_text')
    train_image_da = DocumentArray.load(embed_loc + 'train_image')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    img = preprocess(Image.open(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)

    img_doc = Document(
        mime_type="image/jpeg",
        embedding=image_features.numpy()
    )

    for i, doc in enumerate(train_image_da):
        train_image_da[i].index = i
    result = train_image_da.find(img_doc, limit=5)
    preds = []
    for i in range(5):
        doc_result = result[0][i]
        index = train_image_da[doc_result.id].index
        preds.append(train_text_da[index].text)
    return preds

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