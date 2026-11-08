# Age-Gender-Detection

Estimates the gender and age bucket of the faces in a photo or webcam frame using pretrained deep
learning models in OpenCV's DNN module. The project's motivating use case is to discourage
catfishing on social media by checking that a user's live camera photo plausibly matches the age
and gender on their profile at registration. This repository contains the estimation part only,
not a full verification system.

## How it works

1. **Face detection:** OpenCV's ResNet-10 SSD face detector (`opencv_face_detector_uint8.pb`)
   finds faces above a confidence threshold (default 0.7).
2. **Gender and age:** each face crop, padded by 20 px, goes through the Caffe models from
   Levi & Hassner (2015), trained on the Adience dataset. Gender is Male or Female. Age is one of
   8 buckets: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100).
3. The annotated image shows a box and a `Gender, (age bucket)` label for each face.

## Tech stack

Python 3.10+ (tested with 3.12), OpenCV 4.x (`cv2.dnn`), NumPy, Jupyter/Colab, Matplotlib,
gdown (model download), pytest.

> OpenCV **5** removed the Caffe and TensorFlow model importers, so `requirements.txt` pins
> `opencv-python<5`.

## Setup

```bash
git clone https://github.com/SAMARTHJAISWAL/Age-Gender-Detection.git
cd Age-Gender-Detection
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # requirements.txt is enough to run the CLI
python scripts/download_models.py            # ~94 MB into ./models
```

### Models

`scripts/download_models.py` uses gdown to fetch these six files from the project's
[Google Drive folder](https://drive.google.com/drive/folders/1Ubf32g3G-6cxCl6R5yZMDKlhVSlVXzTI)
into `./models`. Set `AGEGENDER_MODEL_DIR` or pass a path to use another folder.

| File | Size |
|---|---|
| `opencv_face_detector.pbtxt`, `opencv_face_detector_uint8.pb` | 35 KB, 2.7 MB |
| `age_deploy.prototxt`, `age_net.caffemodel` | 2 KB, 46 MB |
| `gender_deploy.prototxt`, `gender_net.caffemodel` | 2 KB, 46 MB |

## Usage

Command line:

```bash
python -m agegender --image photo.jpg                          # prints results, opens a window
python -m agegender --image photo.jpg --output out.jpg --no-display
python -m agegender --camera 0                                # live webcam; ESC or q to quit
```

Example output:

```
Face 1 at (257, 80, 402, 256): Male (1.00), age (25-32) (0.97)
```

Notebook (`Age_and_Gender_Detection_DS-3.ipynb`):

- **Google Colab:** open the notebook in Colab and run all cells. It clones this repo, downloads
  the models, and asks for a webcam photo through the browser.
- **Locally:** `jupyter notebook`, then set `IMAGE_PATH` (or `AGEGENDER_IMAGE`) to a photo, or
  set `USE_LOCAL_WEBCAM = True`.

Tests (pure helpers only, no models needed):

```bash
python -m pytest -q
```

## Project structure

```
Age_and_Gender_Detection_DS-3.ipynb   walkthrough notebook (local + Colab)
agegender/
  core.py        box conversion/padding, label selection (NumPy only, unit-tested)
  detector.py    AgeGenderDetector: loads the nets, predict(), annotate()
  models.py      model file names and location (AGEGENDER_MODEL_DIR)
  __main__.py    CLI for images and webcam
scripts/download_models.py   fetch the model files with gdown
tests/                       pytest suite
```

## Limitations

- The age output is a coarse bucket, and the 2015 Adience models are known to be biased and
  less accurate across demographics, ages and lighting conditions. The output is not reliable
  enough on its own to verify anyone's identity or age.
- Gender is a binary appearance-based estimate.
- Earlier versions of this README mentioned Twitter data collection. No such code exists in the
  repository.
