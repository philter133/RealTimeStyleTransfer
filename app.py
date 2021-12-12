import uuid

import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from itsdangerous import json
from skimage import color

from ImageColorization.Architectures.Generator import Unet
from ImageColorization.utils import lab_to_rgb
from Models.Transformer import TransformerNetwork
from torchvision import transforms
from Database.Mongo import PhilterDB
from PIL import Image
from NSTMain import NeuralStyleTransfer
from VGG import VGGModel
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

    checkpoint = torch.load(path, map_location=device)

    model = TransformerNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_name = i.split(".model")[0]
    model_dict.update({model_name: model})

checkpoint = torch.load("./BWModel/Restorer.model", map_location=device)
bw_model = Unet(256, 64, 0.5, False).to(device)
bw_model.load_state_dict(checkpoint["generator"])

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


def bw_to_color(image_bytes: bytes,
                transformer: transforms.Compose,
                generator: Unet):
    with torch.no_grad():
        generator.eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = transformer(img)
        arr = np.array(img)

        lab = color.rgb2lab(arr).astype("float32")
        ten_lab = transforms.ToTensor()(lab)

        L = ten_lab[0:1] / 50 - 1

        L = torch.unsqueeze(L, dim=0).to(device)

        fake_ab = generator(L)

        fake_ab = torch.squeeze(fake_ab, dim=0)
        L = torch.squeeze(L, dim=0)

        img_rgb = lab_to_rgb(L, fake_ab)

        img_rgb = np.clip(img_rgb, 0, 255)

        im = Image.fromarray(img_rgb)

    return im


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


@app.route('/bw-color', methods=["POST", "GET"])
def convert_to_color():
    transformer_list = [transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(256)]

    file = request.files['file']
    img_bytes = file.read()

    im = bw_to_color(img_bytes,
                     transforms.Compose(transformer_list),
                     bw_model)

    img_io = io.BytesIO()
    im.save(img_io, "JPEG", quality=70)
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
    generated = True if request.form["generated"].upper() == "TRUE" else False

    image_id = str(uuid.uuid4())

    db.save_image(
        image_id,
        title,
        file,
        description=description,
        generated=generated)

    return jsonify({"imageId": image_id})


@app.route('/style-image', methods=["POST", "GET"])
def style_image():
    content_file = request.files['contentImage']
    style_file = request.files['styleImage']
    image_size = int(request.form['imageSize'])
    layer_set = request.form['layerSet']
    content_input = True if request.form['contentInput'].upper() == "TRUE" else False
    style_weight = float(request.form["styleWeight"])
    content_weight = float(request.form["contentWeight"])

    epochs = request.form["epochs"]

    content_bytes = content_file.read()
    style_bytes = style_file.read()

    nst = NeuralStyleTransfer(style_bytes,
                              content_bytes,
                              image_size,
                              layer_set,
                              True,
                              content_input,
                              [1, 1, 1, 1, 1],
                              content_weight,
                              style_weight,
                              0.01)


    for i in range(int(epochs)):
        nst.train_one_adam(100)

    im = nst.get_image()

    img_io = io.BytesIO()
    im.save(img_io, "JPEG", quality=70)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


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


@app.route('/get-image-cluster', methods=["POST", "GET"])
def get_image_cluster():
    user_id = request.form["userId"]
    limit = int(request.form["limit"])
    ascend = True if request.form["ascending"].upper() == "TRUE" else False
    page_num = int(request.form["pageNumber"])

    pack_dict = {}

    algo = request.form["algorithm"].upper()
    if algo != "NONE":
        pack_dict["algorithm"] = algo

    tag = request.form["algorithm"].upper()
    if tag != "NONE":
        pack_dict["tag"] = tag

    clusters = db.get_clusters(user_id,
                               limit,
                               ascend,
                               page_num,
                               **pack_dict)

    tag_list = []
    algo_list = []
    gen_list = []
    base_list = []

    for cluster in clusters["clusters"]:
        tag_list.append(cluster["tag"])
        algo_list.append(cluster["algorithm"])

        data, gen = db.cluster_to_image(cluster["imageList"])

        gen_list.append(gen[0])
        base_list.append(data)

    data_dict = {
        "tagList": tag_list,
        "algoList": algo_list,
        "genList": gen_list,
        "baseList": base_list,
        "nextPage": clusters["next_page"]
    }

    return jsonify(data_dict)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=os.environ.get('PORT', 80))
