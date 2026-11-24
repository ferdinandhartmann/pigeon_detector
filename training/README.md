# Train the bird models

Run the commands from the root of the Pigeon checkout in WSL. Create the laptop environment in `.venv/` if it does not already exist.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install Flask Pillow PyYAML 'ultralytics==8.4.82' 'tensorflow[and-cuda]' -e './pigeon_detector[tflite]'
nvidia-smi
```

For Google Drive captures, follow the [download instructions](../../device/model_tools/README.md#download-captures-from-google-drive) first.

The reviewer imports images and metadata from `device-data/` into `captures/`. Existing reviews are preserved. Pass `--device-data /path/to/downloads` to use a different source.

```bash
.venv/bin/python pigeon_detector/training/workflow.py review
```

Visit http://localhost:5000 in Windows. Correct boxes and approve each frame, including empty scenes. Use Uncertain for birds whose identity you cannot establish. Stop the reviewer with Ctrl+C. Back up the captures directory after reviewing.

```bash
.venv/bin/python pigeon_detector/training/workflow.py prepare dataset-002
# The commands below use the existing dataset-001. Choose a fresh run name to retrain.
.venv/bin/python pigeon_detector/training/workflow.py train-detector dataset-001 run2
.venv/bin/python pigeon_detector/training/workflow.py train-classifier dataset-001 run2
.venv/bin/python pigeon_detector/training/workflow.py export dataset-001 run2
```

Classifier training needs both labels in the training split. With validation crops, the classifier saves the checkpoint with the lowest validation loss. Without validation crops, it trains on the training split and saves the final epoch. Evaluation needs a populated test split. Capture groups use the UTC capture day or, when timestamps are missing, the source folder. For updates, reuse the same working directory and choose fresh dataset and run names.

Edit [config.yaml](config.yaml) for model and training settings. Both train commands load it automatically, and `--epochs` overrides the YAML for one run.

Datasets go under `datasets/` and training runs under `runs/` by default. Use `prepare dataset-002 --include-test-photos` to include reviewed device test photos in a fresh output folder. Detector training requires validation images.

## Copy the bundle to the Pi

```bash
bash pigeon_detector/training/copy_to_pi.sh run2
```

The script checks the bundle under `runs/run2/bundle/` and copies it over SSH to `/home/ferdi/pigeon/pigeon_detector/data/models/pigeon-models-dataset-001-run2/` on the Pi. To select a bundle later, update the detector and classifier model and label paths in the Pi's `config.yaml`.
