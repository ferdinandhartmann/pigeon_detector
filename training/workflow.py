"""Laptop workflow. Run commands from a dedicated working directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path("captures")
CLASSES = ["Other bird", "Pigeon/Dove", "Uncertain"]


def review():
    from flask import Flask, request, send_from_directory, jsonify
    app = Flask(__name__)
    @app.get("/")
    def index():
        return Path(__file__).with_name("review.html").read_text()
    @app.get("/records")
    def records():
        return jsonify(sorted(p.stem for p in ROOT.glob("*.json")))
    @app.get("/files/<name>")
    def file(name):
        return send_from_directory(ROOT.resolve(), name)
    @app.post("/save/<name>")
    def save(name):
        path = ROOT / (name + ".json")
        data = json.loads(path.read_text())
        boxes = request.json["boxes"]
        with Image.open(ROOT / data["image"]) as im:
            for x,y,r,b,label in boxes:
                if not (0 <= x < r <= im.width and 0 <= y < b <= im.height and label in range(3)):
                    return "Correct or delete the highlighted suggestions before approving.", 400
        data.update(boxes=boxes, reviewed=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        return "Saved"
    app.run(host="127.0.0.1", threaded=False)


def prepare(out):
    import yaml
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(ROOT.glob("*.json")):
        d = json.loads(path.read_text())
        if not d["reviewed"] or d["test_photo"]:
            continue
        manifest[path.stem] = d
        split = d["split"]
        image = ROOT / d["image"]
        images = out / "images" / split
        labels = out / "labels" / split
        for directory in [images, labels]:
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, images / image.name)
        lines = []
        with Image.open(image).convert("RGB") as frame:
            w,h = frame.size
            for i,(x,y,r,b,c) in enumerate(d["boxes"]):
                lines.append(f"0 {(x+r)/(2*w)} {(y+b)/(2*h)} {(r-x)/w} {(b-y)/h}")
                if c < 2:
                    dest = out / "crops" / split / str(c)
                    dest.mkdir(parents=True, exist_ok=True)
                    frame.crop((int(x),int(y),max(int(x)+1,int(r)),max(int(y)+1,int(b)))).save(dest / f"{path.stem}-{i}.png")
        (labels / path.name).with_suffix(".txt").write_text("\n".join(lines))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"birds.yaml").write_text(yaml.safe_dump(dict(path=str(out.resolve()),train="images/train",val="images/val",test="images/test",names=["bird"])))


def classifier(data, run, epochs):
    import tensorflow as tf
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow cannot see the RTX GPU")
    def dataset(split):
        ds = tf.keras.utils.image_dataset_from_directory(data/"crops"/split, class_names=["0","1"], image_size=(224,224),batch_size=16)
        return ds.map(lambda x,y: ((x-128.)/128.,y))
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False,pooling="avg")
    base.trainable = False
    inp = tf.keras.Input((224,224,3))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2,activation="softmax")(base(inp,training=False)))
    model.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])
    run.mkdir(parents=True,exist_ok=True)
    model.fit(dataset("train"),validation_data=dataset("val"),epochs=epochs,callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(run/"classifier.keras"),save_best_only=True)])


def export(data,run):
    import tensorflow as tf
    import numpy as np
    from ultralytics import YOLO
    bundle = run/"bundle"
    bundle.mkdir(exist_ok=True)
    exported = Path(YOLO(run/"detector/weights/best.pt").export(format="tflite",int8=True,data=str(data/"birds.yaml")))
    files = list(exported.parent.glob("*full_integer_quant.tflite"))
    if len(files)!=1:
        raise RuntimeError(f"Expected one full integer detector in {exported.parent}")
    shutil.copyfile(files[0],bundle/"detector.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(run/"classifier.keras"))
    def examples():
        for path in sorted((data/"crops/train").glob("*/*.png"))[::max(1,len(list((data/"crops/train").glob("*/*.png")))//300)]:
            with Image.open(path) as im:
                yield [(np.asarray(im.resize((224,224)),dtype=np.float32)[None]-128)/128]
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=examples
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.uint8
    converter.inference_output_type=tf.uint8
    (bundle/"classifier.tflite").write_bytes(converter.convert())
    (bundle/"objects.txt").write_text("bird\n")
    (bundle/"species.txt").write_text("\n".join(CLASSES[:2])+"\n")
    (bundle/"manifest.json").write_text(json.dumps(dict(dataset=data.name,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.name!="manifest.json"})))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest="cmd",required=True)
    for cmd in ["review","prepare","train-detector","train-classifier","export"]:
        sub=subs.add_parser(cmd)
        if cmd!="review":
            sub.add_argument("data",type=Path)
        if cmd.startswith("train") or cmd=="export":
            sub.add_argument("run",type=Path)
        if cmd.startswith("train"):
            sub.add_argument("--epochs",type=int,default=25)
    a=p.parse_args()
    if a.cmd=="review": review()
    elif a.cmd=="prepare": prepare(a.data)
    elif a.cmd=="train-classifier": classifier(a.data,a.run,a.epochs)
    elif a.cmd=="export": export(a.data,a.run)
    else:
        from ultralytics import YOLO
        YOLO("yolo26n.pt").train(data=str(a.data/"birds.yaml"),device=0,batch=4,epochs=a.epochs,project=str(a.run.resolve()),name="detector",imgsz=640)
