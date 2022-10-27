import io
import os
import numpy
from PIL import Image
from flask import Flask,  render_template, request, redirect
import torch
app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def get_prediction(img_bytes):
    model = torch.hub.load('./temp/ultralytics_yolov5_master', 'custom', path='best.pt', source='local')
    model.eval()
    model.conf = 0.5

    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        # print(results.xyxyn, file=sys.stderr)

        labels = results.xyxyn[0][:, -1].numpy()
        # print(results, file=sys.stderr)
        # print(labels, file=sys.stderr)
        # results.save(save_dir='static')

        # full_filename = os.path.join(app.config['RESULT_FOLDER'], 'results0.jpg')
        return "Class no " + numpy.array2string(labels)
    else:
        if len(os.listdir('temp')) == 0:
            torch.hub.set_dir('temp')
            torch.hub.load("ultralytics/yolov5", 'custom', path="best.pt")
        return render_template('index.html')
