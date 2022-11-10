import io
import os
import numpy
from PIL import Image
from flask import Flask,  render_template, request, redirect
import torch
app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if __name__ == "__main__":
    app.run()


def get_prediction(img_bytes):
    model = torch.hub.load('./temp/ultralytics_yolov5_master', 'custom', path='best.pt', source='local')
    model.cpu()
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

        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        print(cord_thres)

        l = numpy.array2string(labels)
        t = numpy.array2string(cord_thres)
        res = []
        for index, label in enumerate(labels):
            res.append([label, cord_thres[index][0], cord_thres[index][1], cord_thres[index][2], cord_thres[index][3], cord_thres[index][4]])

        return str(res)
    else:
        if len(os.listdir('temp')) == 1:
            torch.hub.set_dir('temp')
            torch.hub.load("ultralytics/yolov5", 'custom', path="best.pt")
        return render_template('index.html')
