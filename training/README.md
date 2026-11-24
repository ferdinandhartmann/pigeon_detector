# Train the bird models

Use Python 3.11 in Ubuntu under WSL2. Confirm that `nvidia-smi` lists the RTX 4060. From the detector checkout, install the training dependencies.

```bash
python3.11 -m venv ~/bird-env
source ~/bird-env/bin/activate
pip install Flask Pillow PyYAML 'ultralytics==8.4.82' 'tensorflow[and-cuda]' -e '.[tflite]'
python -c 'import torch,tensorflow as tf; assert torch.cuda.is_available() and tf.config.list_physical_devices("GPU")'
```

Use the Pi project's capture importer to populate `~/bird-training/captures`. The commands below assume the detector checkout is at `~/pigeon/pigeon_detector`.

```bash
cd ~/bird-training
python ~/pigeon/pigeon_detector/training/workflow.py review
```

Visit http://localhost:5000 in Windows. Correct boxes and approve each frame, including empty scenes. Use Uncertain for birds whose identity you cannot establish. Stop the reviewer with Ctrl+C. Back up the captures directory after reviewing.

```bash
python ~/pigeon/pigeon_detector/training/workflow.py prepare dataset-001
python ~/pigeon/pigeon_detector/training/workflow.py train-detector dataset-001 run-001
python ~/pigeon/pigeon_detector/training/workflow.py train-classifier dataset-001 run-001
python ~/pigeon/pigeon_detector/training/workflow.py export dataset-001 run-001
```

Training needs both classifier labels in the training and validation splits. Evaluation needs a populated test split. Capture groups use the UTC capture day or, when timestamps are missing, the source folder. For updates, reuse the same working directory and choose fresh dataset and run names.
