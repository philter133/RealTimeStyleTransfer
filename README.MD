# Welcome to Philter

For the frontend source code click this link [github.com](https://github.com/philter133).
For the backend source code click this link [github.com](https://github.com/philter133/RealTimeStyleTransfer/tree/develop).

## Running the project
Frontend Requirements
- npm must be installed

In order to start the frontend server make sure to unzip the frontend project
then install the project with these lines

```bash
npm install 
npm install next
npm install react-google-login
```

In order to start the frontend server make sure to unzip the frontend project
then install the project with these lines

from there simply run it with these lines

```bash
npm run dev
```

Backend Requirements
- 3.10 > python > 3.7

In order to start the Backend server make sure to unzip the Backend project
then install the project with these lines

```bash
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt
```

from there simply run it with these lines

```bash
flask run
```

Now the project is ready to be ran

All dependencies are in the requirements.txt file

Documentation
------------------------------------------------------------
Apply Filter, applies a filter to the image, based on what filter the user wants
```python
@app.route('/apply-filter', methods=['POST', 'GET'])
def apply_filter():
    file = request.files['file']
    name = request.form['name']
    size = request.form['size']
    title = request.form['title']
    description = request.form['description']

    image_id_list = []

    size_dict = {"micro": 500,
                 "small": 750,
                 "medium": 1000,
                 "large": 1500,
                 "mega": 2250}

    if size.lower() not in list(size_dict.keys()):
        flask.abort(500)

    image_id = str(uuid.uuid4())
    img_bytes = file.read()
    image = transform_image(img_bytes,
                            model_dict[name],
                            size_dict[size])

    img_byte_arr = io.BytesIO()

    image.save(img_byte_arr, format='JPEG')

    inserted_id, gen = db.save_image(image_id,
                                     title,
                                     img_byte_arr.getvalue(),
                                     description=description,
                                     generated=True)
    image_id_list.append(inserted_id)

    image_id = str(uuid.uuid4())
    file.stream.seek(0)
    img_bytes = file.read()
    img_io = io.BytesIO(img_bytes)
    img = Image.open(img_io)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_io.seek(0)

    inserted_id, _ = db.save_image(image_id,
                                   "Content Image",
                                   img_byte_arr.getvalue(),
                                   description="The image the style was applied on",
                                   generated=False)

    image_id_list.append(inserted_id)

    return jsonify({"genId": image_id_list[0],
                    "contentId": image_id_list[1],
                    "displayUrl": gen})
```

Turns a black and white image into a color image

```python
@app.route('/bw-color', methods=["POST", "GET"])
def convert_to_color():
    transformer_list = [transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(256)]

    file = request.files['file']
    title = request.form['title']
    description = request.form['description']

    image_id_list = []

    img_bytes = file.read()

    im = bw_to_color(img_bytes,
                     transforms.Compose(transformer_list),
                     bw_model)

    img_io = io.BytesIO()
    im.save(img_io, "JPEG", quality=70)

    image_id = str(uuid.uuid4())
    inserted_id, gen = db.save_image(image_id,
                                     title,
                                     img_io.getvalue(),
                                     description=description,
                                     generated=True)

    image_id_list.append(inserted_id)

    image_id = str(uuid.uuid4())
    file.stream.seek(0)
    img_bytes = file.read()
    img_io = io.BytesIO(img_bytes)
    img = Image.open(img_io)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_io.seek(0)

    inserted_id, _ = db.save_image(image_id,
                                   "Content Image",
                                   img_byte_arr.getvalue(),
                                   description="The image the style was applied on",
                                   generated=False)

    image_id_list.append(inserted_id)

    return jsonify({"genId": image_id_list[0],
                    "contentId": image_id_list[1],
                    "displayUrl": gen})

```
Applies a Style to an image based on the style image 
provided

```python
@app.route('/style-image', methods=["POST", "GET"])
def style_image():
    content_file = request.files['contentImage']
    style_file = request.files['styleImage']

    image_size = int(request.form['imageSize'])
    layer_set = request.form['layerSet']
    style_weight = float(request.form["styleWeight"])
    content_weight = float(request.form["contentWeight"])

    print(layer_set)

    epochs = request.form["epochs"]
    title = request.form["title"]
    description = request.form["description"]

    content_bytes = content_file.read()
    style_bytes = style_file.read()

    image_id_list = []

    nst = NeuralStyleTransfer(style_bytes,
                              content_bytes,
                              image_size,
                              "e",
                              True,
                              True,
                              [1, 1, 1, 1, 1],
                              content_weight,
                              style_weight,
                              0.01)

    for z in range(int(epochs)):
        print(z)
        nst.train_one_adam(100)

    im = nst.get_image()

    img_io = io.BytesIO()
    im.save(img_io, "JPEG", quality=70)

    image_id = str(uuid.uuid4())
    inserted_id, gen = db.save_image(image_id,
                                     title,
                                     img_io.getvalue(),
                                     description=description,
                                     generated=True)

    image_id_list.append(inserted_id)

    image_id = str(uuid.uuid4())
    content_file.stream.seek(0)
    img_bytes = content_file.read()
    img_io = io.BytesIO(img_bytes)
    img = Image.open(img_io)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_io.seek(0)

    inserted_id, _ = db.save_image(image_id,
                                   "Content Image",
                                   img_byte_arr.getvalue(),
                                   description="The image the style was applied on",
                                   generated=False)

    image_id_list.append(inserted_id)

    image_id = str(uuid.uuid4())
    style_file.stream.seek(0)
    img_bytes = style_file.read()
    img_io = io.BytesIO(img_bytes)
    img = Image.open(img_io)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_io.seek(0)

    inserted_id, _ = db.save_image(image_id,
                                   "Style Image",
                                   img_byte_arr.getvalue(),
                                   description="The style applied on the content",
                                   generated=False)

    image_id_list.append(inserted_id)

    return jsonify({"genId": image_id_list[0],
                    "contentId": image_id_list[1],
                    "styleId": image_id_list[2],
                    "displayUrl": gen})

```