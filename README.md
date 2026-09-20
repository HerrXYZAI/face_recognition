# face_recognition

A pipeline that learns to recognize specific people from faces already tagged
in a Lightroom Classic catalog, then applies that to an entire photo library
and writes the results back so Lightroom can display them.

Four steps, each its own CLI subcommand:

1. **`export-faces`** — export person-tagged faces (name + bounding box)
   from your Lightroom Classic catalog.
2. **`train-classifier`** — build a face-recognition model for your specific
   people, using step 1's output as labeled training data.
3. **`run-inference`** — run that model across your whole photo library,
   storing every detected face + bounding box in a local database.
4. **`write-xmp`** — write the results back out as XMP face regions so
   Lightroom can show them.

See [`plans` in this repo's history](.) — design rationale and the tradeoffs
behind each step live as comments in the corresponding module.

## How it works

- **Step 1** reads a *copy* of your `.lrcat` catalog file directly (it's an
  undocumented but plain SQLite database) — no Lightroom plugin needed, and
  Lightroom doesn't need to be running. See
  [`face_pipeline/lightroom/catalog_export.py`](face_pipeline/lightroom/catalog_export.py).
- **Step 2** uses a pretrained face embedding model
  ([InsightFace](https://github.com/deepinsight/insightface), `buffalo_l` /
  ArcFace) rather than training a network from scratch, and fits a
  lightweight classifier (linear SVM) on top using your Lightroom-tagged
  faces. This is far more practical than fine-tuning the embedding network
  itself given how much labeled data personal photo libraries typically have
  per person. See
  [`face_pipeline/recognition/train_classifier.py`](face_pipeline/recognition/train_classifier.py).
- **Step 3** walks your image library, detects + classifies every face, and
  stores results (including raw embeddings, so the classifier can be
  retrained later without redoing detection) in a local SQLite database
  (`data/faces.db`) — entirely separate from Lightroom's own catalog, so
  nothing here can corrupt it. Resumable: re-running after adding new photos
  only processes what changed. See
  [`face_pipeline/pipeline/run_inference.py`](face_pipeline/pipeline/run_inference.py).
- **Step 4** writes face regions back out using the Metadata Working Group
  (MWG) `RegionInfo` XMP structure — embedded directly for formats Lightroom
  itself embeds into (DNG/JPEG/TIFF/PSD), or as an external `.xmp` sidecar
  for proprietary raw formats, matching Lightroom's own convention. See
  [`face_pipeline/lightroom/xmp_writer.py`](face_pipeline/lightroom/xmp_writer.py).

  **Manual finishing step required in Lightroom:** Lightroom only
  auto-imports MWG face regions from XMP on *import* of new files. For
  photos already in your catalog, after running `write-xmp` you need to
  select the affected photos in Lightroom and run
  *Metadata → Read Metadata from Files* so the new regions and names appear.
  There's no reliable way to trigger this from outside Lightroom.

## Setup (Windows, native — `run.bat`)

Double-click [`run.bat`](run.bat), or run it from a terminal. On first run it:
- finds or bootstraps Conda (Anaconda/Miniconda) and creates a `face_pipeline`
  conda environment (Python 3.11), or falls back to a local `.venv` if no
  Conda install is found;
- installs all dependencies into it (GPU-enabled `onnxruntime-gpu` if an
  NVIDIA GPU is detected via `nvidia-smi`, otherwise CPU-only `onnxruntime`);
- checks for `exiftool` (needed only for the final write-back step) — either
  on `PATH` or as `exiftool.exe` bundled directly next to `run.bat`;
- creates `config.yaml` from `config.example.yaml` and opens it in Notepad
  for you to fill in.

It then opens a menu to run each step (inspect the catalog, export, verify
crops, train, run inference, check status, write XMP back), so you can
re-run individual steps as needed. Re-running `run.bat` later skips the
setup steps (env/deps already in place) and goes straight to the menu.

## Setup (Docker — alternative)

Requires [Docker](https://docs.docker.com/get-docker/), and for GPU
acceleration the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

```sh
cp config.example.yaml config.yaml     # then edit the paths, see comments inside
cp docker/.env.example docker/.env     # then edit IMAGES_ROOT_HOST / CATALOG_COPY_DIR_HOST

docker compose -f docker/docker-compose.yml --profile gpu build
# no NVIDIA GPU? use --profile cpu instead, and the pipeline-cpu service below

# 1. Make a COPY of your Lightroom catalog (Lightroom locks the live file),
#    put it in the folder you set as CATALOG_COPY_DIR_HOST, then:
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu inspect-catalog
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu export-faces

# 2. Sanity-check a handful of exported crops before trusting the export:
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu verify-crops
#    -> open data/verify_crops/*.jpg and confirm each shows the named person

# 3. Train, then test on a small sample before committing to the full library:
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu train-classifier
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu run-inference --limit 200
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu status

# 4. Once happy, run the full library, then write results back to Lightroom:
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu run-inference
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu write-xmp --dry-run
docker compose -f docker/docker-compose.yml --profile gpu run --rm pipeline-gpu write-xmp
```

Then in Lightroom: select the affected photos → *Metadata → Read Metadata
from Files*.

## Setup (without Docker)

```sh
pip install -e ".[dev]"
pip install onnxruntime-gpu   # or: pip install onnxruntime   for CPU-only
# install exiftool from https://exiftool.org and ensure it's on PATH
cp config.example.yaml config.yaml   # edit paths for your machine
face-pipeline export-faces
# ... same subcommands as above, without the docker compose wrapper
```

## Notes / known limitations

- Lightroom's catalog schema is undocumented and has changed across
  versions. `catalog_export.py` inspects the actual schema at runtime and
  fails with a clear error (listing the real column names) if it doesn't
  recognize the bounding-box columns, rather than guessing silently — run
  `face-pipeline inspect-catalog` first against your real catalog copy.
- Face bounding boxes are assumed to be relative to the EXIF-corrected
  (displayed) image orientation. Run `face-pipeline verify-crops` to
  visually confirm this against your own catalog before a full training run.
- Nothing in this project ever writes into your Lightroom `.lrcat` catalog —
  only into a separate local `faces.db` and, in step 4, XMP metadata in your
  photo files/sidecars.

## Tests

```sh
pip install -e ".[dev]"
pytest
```

`tests/test_xmp_writer.py`'s round-trip test is skipped automatically if
`exiftool` isn't installed on the machine running the tests.
