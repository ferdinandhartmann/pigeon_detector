"""Laptop workflow. Run commands from a dedicated working directory."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("captures")
REPO = Path(__file__).resolve().parents[2]
CLASSES = ["Other bird", "Pigeon/Dove", "Uncertain"]


def review(device_data):
    if device_data.is_dir():
        importer = REPO / "device" / "model_tools" / "import_captures.py"
        subprocess.run([sys.executable, str(importer), str(device_data)], check=True)
    elif device_data != REPO / "device-data":
        raise SystemExit(f"Capture folder does not exist: {device_data}")

    from flask import Flask, request, send_from_directory, jsonify
    app = Flask(__name__)
    @app.get("/")
    def index():
        return Path(__file__).with_name("review.html").read_text()
    @app.get("/records")
    def records():
        paths = sorted(ROOT.glob("*.json"))
        if request.args.get("include_status") == "1":
            return jsonify([
                {"id": path.stem, "reviewed": bool(json.loads(path.read_text()).get("reviewed"))}
                for path in paths
            ])
        return jsonify([path.stem for path in paths])
    @app.get("/files/<name>")
    def file(name):
        return send_from_directory(ROOT.resolve(), name)
    @app.delete("/records/<name>")
    def delete_capture(name):
        record = ROOT / (name + ".json")
        try:
            if not record.exists():
                return "", 204
            filename = json.loads(record.read_text())["image"]
            image = Path(filename)
            if image.name != filename or image.stem != name or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                return jsonify(error="The record has an invalid image filename."), 400
            # Remember deletions when the same device folder is imported again.
            deleted = ROOT / ".deleted"
            deleted.mkdir(exist_ok=True)
            (deleted / record.name).touch()
            (ROOT / image).unlink(missing_ok=True)
            record.unlink()
        except (ValueError, KeyError, TypeError):
            return jsonify(error="The capture record has invalid image metadata."), 400
        except OSError as exc:
            return jsonify(error=f"Could not delete {exc.filename}: {exc.strerror}"), 500
        return "", 204

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


def prepare(out, include_test_photos=False):
    import yaml
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    unreviewed = skipped_test_photos = 0
    for path in sorted(ROOT.glob("*.json")):
        d = json.loads(path.read_text())
        if not d["reviewed"]:
            unreviewed += 1
            continue
        if d["test_photo"] and not include_test_photos:
            skipped_test_photos += 1
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
    print(f"Prepared {len(manifest)} images in {out}. Skipped {unreviewed} unreviewed captures and {skipped_test_photos} reviewed test photos.")
    if skipped_test_photos:
        print("Use --include-test-photos with a fresh output directory to include reviewed test photos.")


def classifier(data, run, settings):
    import tensorflow as tf
    if settings["require_gpu"] and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow cannot see the RTX GPU")
    image_size = tuple(settings["image_size"])

    def dataset(split):
        ds = tf.keras.utils.image_dataset_from_directory(data/"crops"/split, class_names=["0","1"], image_size=image_size,batch_size=settings["batch_size"])
        return ds.map(lambda x,y: ((x-128.)/128.,y))
    train = dataset("train")
    val_dir = data / "crops" / "val"
    validation = None
    if any(val_dir.glob("*/*.png")):
        if not all((val_dir / label).is_dir() for label in ("0", "1")):
            raise ValueError("Validation crops must have directories for both classifier labels (0 and 1).")
        validation = dataset("val")
    else:
        print("No validation crops found. Training on all training crops and saving the final epoch.")
    backbone = getattr(tf.keras.applications, settings["model"], None)
    if not callable(backbone):
        raise ValueError(f'Unknown Keras Applications model {settings["model"]!r}')
    input_shape = (*image_size, 3)
    base = backbone(input_shape=input_shape, weights=settings["weights"], include_top=False, pooling="avg")
    base.trainable = settings["train_backbone"]
    inp = tf.keras.Input(input_shape)
    features = base(inp, training=None if base.trainable else False)
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2, activation="softmax")(features))
    model.compile(settings["optimizer"], "sparse_categorical_crossentropy", metrics=["accuracy"])
    run.mkdir(parents=True,exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(run / "classifier.keras"), save_best_only=validation is not None,
        monitor="val_loss" if validation is not None else "loss",
    )
    model.fit(train, validation_data=validation, epochs=settings["epochs"], callbacks=[checkpoint])


def export(data,run, detector_file=None):
    import tensorflow as tf
    import numpy as np
    from ultralytics import YOLO
    bundle = run/"bundle"
    bundle.mkdir(exist_ok=True)
    if detector_file is None:
        exported = Path(YOLO(run/"detector/weights/best.pt").export(format="tflite",int8=True,data=str(data/"birds.yaml")))
        files = list(exported.parent.glob("*full_integer_quant.tflite"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one full integer detector in {exported.parent}")
        detector_file = files[0]
    detector_target = bundle / "detector.tflite"
    if detector_file.resolve() != detector_target.resolve():
        shutil.copyfile(detector_file, detector_target)
    model = tf.keras.models.load_model(run / "classifier.keras")
    height, width = model.input_shape[1:3]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    def examples():
        for path in sorted((data/"crops/train").glob("*/*.png"))[::max(1,len(list((data/"crops/train").glob("*/*.png")))//300)]:
            with Image.open(path) as im:
                yield [(np.asarray(im.resize((width, height)),dtype=np.float32)[None]-128)/128]
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=examples
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.uint8
    converter.inference_output_type=tf.uint8
    (bundle/"classifier.tflite").write_bytes(converter.convert())
    (bundle/"objects.txt").write_text("bird\n")
    (bundle/"species.txt").write_text("\n".join(CLASSES[:2])+"\n")
    (bundle/"manifest.json").write_text(json.dumps(dict(dataset=data.name,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.name!="manifest.json"})))


def output_path(value, directory, *, existing=False):
    path = Path(value)
    if "/" in value:
        return path
    default = Path(directory) / path
    if existing and not default.exists() and path.exists():
        return path
    return default


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest="cmd",required=True)
    for cmd in ["review","prepare","train-detector","train-classifier","export"]:
        sub=subs.add_parser(cmd)
        if cmd=="review":
            sub.add_argument("--device-data", type=Path, default=REPO / "device-data",
                             help="source folder for downloaded captures")
        else:
            sub.add_argument("data", help="dataset name under datasets/ or an explicit path")
        if cmd=="prepare":
            sub.add_argument("--include-test-photos", action="store_true", help="include reviewed device test photos")
        if cmd.startswith("train") or cmd=="export":
            sub.add_argument("run", help="run name under runs/ or an explicit path")
        if cmd.startswith("train"):
            sub.add_argument("--epochs", type=int, help="override config.yaml epochs")
        if cmd=="export":
            sub.add_argument("--detector-file", type=Path, help="reuse an existing TFLite detector")
    a=p.parse_args()
    if a.cmd != "review":
        a.data = output_path(a.data, "datasets", existing=a.cmd != "prepare")
    if hasattr(a, "run"):
        a.run = output_path(a.run, "runs", existing=a.cmd == "export")
    if a.cmd=="review": review(a.device_data)
    elif a.cmd=="prepare": prepare(a.data, a.include_test_photos)
    elif a.cmd=="export": export(a.data,a.run,a.detector_file)
    else:
        import yaml

        config_path = Path(__file__).with_name("config.yaml")
        config = yaml.safe_load(config_path.read_text())
        settings = config[a.cmd.removeprefix("train-")]
        if a.epochs is not None:
            settings["epochs"] = a.epochs
        if a.cmd == "train-classifier":
            classifier(a.data, a.run, settings)
        else:
            from ultralytics import YOLO

            weights = settings.pop("model")
            YOLO(weights).train(
                data=str(a.data / "birds.yaml"), project=str(a.run.resolve()),
                name="detector", **settings,
            )
