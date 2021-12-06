import uuid

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from itsdangerous import json

from Models.Transformer import TransformerNetwork
from torchvision import transforms
from Database.Mongo import PhilterDB
from PIL import Image
import torch
import os
import io
import PIL
import flask
import flask_pymongo

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

"""
MODELS
"""
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device=device_name)

model_dict = {}
for i in os.listdir("./SavedModels"):
    path = os.path.join(".", "SavedModels", i)

    if device_name == "cpu":
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)

    model = TransformerNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_name = i.split(".model")[0]
    model_dict.update({model_name: model})

"""
DATABASE
"""

db = PhilterDB()

"""
Feeds image into model returning the filtered image

image_bytes: The image in bytes

filter_model: the model we are using to add the filter to the image

size: The size of the image in pixels

returns the PIL image object
"""


def transform_image(image_bytes: bytes,
                    filter_model: torch.nn.Module,
                    size: int) -> PIL.Image.Image:
    transformer_list = [transforms.Resize(size),
                        transforms.CenterCrop(size),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255))]

    pic_tf = transforms.Compose(transformer_list)
    img = Image.open(io.BytesIO(image_bytes))

    with torch.no_grad():
        filter_model.eval()

        img_tensor = torch.unsqueeze(pic_tf(img), dim=0).to(device)

        img_tensor = img_tensor[:, :3, :, :]  # For png images makes sure it only uses three channels
        output = torch.squeeze(filter_model(img_tensor), dim=0).cpu()

        output = output.detach().clone().numpy()

        filtered = output.transpose(1, 2, 0).clip(0, 255).astype("uint8")
        filtered = Image.fromarray(filtered)

        return filtered


"""
Applies a filter to a specified image

file_path: where the image file is located must be of the type
.png, .jpg, or .jpeg. A string

filter_name: the name of the filter we want to apply to the image.
Options are cudi, edtaonisl, mosaic, scream, starrynight, and muse. A string

size: the size of the new filtered image, the size must be less than or equal too
the actual size of the image. Options are micro, small, medium, large, or mega. A string

returns the image in bytes
"""


@app.route('/apply-filter', methods=['POST', 'GET'])
def apply_filter():
    file = request.files['file']
    name = request.form['name']
    size = request.form['size']

    img_bytes = file.read()

    size_dict = {"micro": 500,
                 "small": 750,
                 "medium": 1000,
                 "large": 1500,
                 "mega": 2250}

    if size.lower() not in list(size_dict.keys()):
        flask.abort(500)

    image = transform_image(img_bytes,
                            model_dict[name],
                            size_dict[size])

    img_io = io.BytesIO()
    image.save(img_io, "JPEG", quality=70)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/login', methods=["POST", "GET"])
def login():
    email = request.form['email']
    name = request.form["name"]

    match_count = db.login_user(email,
                                name)

    return jsonify({"registered": False} if match_count == 1 else {"registered": True})


@app.route('/save-image', methods=["POST", "GET"])
def save_image():
    title = request.form["title"].upper()
    description = request.form["description"]
    file = request.files["file"]

    image_id = str(uuid.uuid4())

    db.save_image(
        image_id,
        title,
        file,
        description=description)

    return jsonify({"imageId": image_id})


@app.route('/save-cluster', methods=["POST", "GET"])
def save_image_cluster():
    user_id = request.form["userId"]
    tag = request.form["tag"].upper()
    image_list = json.loads(request.form["imageList"])
    algorithm_type = request.form["algorithm"].upper()

    inserted_id = db.save_cluster(user_id,
                                  image_list,
                                  algorithm_type,
                                  tag)

    return jsonify({"status": 200,
                    "inserted_id": str(inserted_id)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 80))
