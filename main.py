from flask import Flask, render_template, request, session
from torchvision import transforms
from PIL import Image
from docarray import DocumentArray, Document
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import finetuner
import sys
import torch
import clip
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = '123456789'

# Set the OAuth scope of the Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Set the path to your client_secret.json file
CLIENT_SECRET_FILE = 'client_secret.json'

creds = None
# Create the flow object and run the authentication flow
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

service = build('drive', 'v3', credentials=creds)

def model_predict(image, k, no_image):
    # embed_loc = 'finetune-mimic-clip-0/'
    embed_loc = 'finetune-mimic-clip-4/'
    train_text_da = DocumentArray.load(embed_loc + 'train_text')
    train_image_da = DocumentArray.load(embed_loc + 'train_image')

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # img = preprocess(Image.open(image)).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     image_features = model.encode_image(img)
    # img_doc = Document(
    #     mime_type="image/jpeg",
    #     embedding=image_features.numpy()
    # )

    artifact = "medvision-3-artifact.zip"
    clip_image_encoder = finetuner.get_model(artifact=artifact, select_model="clip-vision")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # _, preprocess = clip.load("ViT-B/32", device=device)
    # img = Image.open('static/uploads/c705d6ea-726251a0-d4ea7c10-6aca1b6e-d016026f-orig.jpg')
    img = DocumentArray([Document(uri=image)])
    image_features = finetuner.encode(model=clip_image_encoder, data=img)

    for i, doc in enumerate(train_image_da):
        train_image_da[i].index = i

    result = train_image_da.find(image_features[0], limit=k)
    preds = None
    if no_image == 'xray':
        preds = "" 
    else:
        preds = "#"
    os.system("rm -rf static/downloads && mkdir static/downloads")
    for i in range(k):
        doc_result = result[0][i]
        index = train_image_da[doc_result.id].index
        preds += (str(train_text_da[index].text) + "@")

        if (no_image == 'xray'):
            # Set the file path of the image you want to download
            file_path = f"project/{train_image_da[doc_result.id].uri}"

            # Search for the file with the given file path
            # project/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10064049/s59114682/a3718ffe-31aeaac6-9a25ee84-e7b030a2-3d0c4bd2.jpg
            file_path = file_path.split("/")[6:]
            PARENT_FOLDER_ID = '1-enZVCmyuNkzcVjhAmMucPyobYnEDLOf'
            for FILE_NAME in file_path:
                query = "name='{}' and parents='{}' and trashed=false".format(FILE_NAME, PARENT_FOLDER_ID)
                files = service.files().list(q=query, fields='files(id)').execute()
                file_id = None
                if files['files']:
                    file_id = files['files'][0]['id']   
                else:
                    print('No file found with name {} in folder {}'.format(FILE_NAME, PARENT_FOLDER_ID))
                PARENT_FOLDER_ID = file_id
            file_id = PARENT_FOLDER_ID
            request = service.files().get_media(fileId=file_id)
            image = Image.open(io.BytesIO(request.execute()))
            image.save(f'static/downloads/image_{i}.jpg')

    return preds[:-1]

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/',  methods=("POST", "GET"))
def fetch_image():
    if request.method == 'POST':
        # os.system('rm -rf static/uploads && mkdir static/uploads')
        upload = request.files['uploaded-file']
        img_name = upload.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        upload.save(img_path)
        session['img_path'] = img_path
    return render_template('index2.html', user_image = img_path)

@app.route('/predict',  methods=("POST", "GET"))
def predict():
    if request.method == "POST":
        kvalue = int(request.form.get("kvalue"))
        no_image = request.form.get("category2")
    image = session.get('img_path', None)
    text = model_predict(image, kvalue, no_image)
    return render_template('index3.html', user_image = image, user_text = text)

if __name__=='__main__':
    app.run(debug = True, host='0.0.0.0', port=80)