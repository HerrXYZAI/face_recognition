"""Single CLI entrypoint for the face pipeline. One subcommand per step:

    face-pipeline inspect-catalog        # dump Lightroom catalog schema (debug)
    face-pipeline export-faces           # step 1
    face-pipeline export-status          # summary of step 1's labeled_faces.csv
    face-pipeline verify-crops           # debug: sanity-check bbox alignment
    face-pipeline train-classifier       # step 2
    face-pipeline run-inference          # step 3
    face-pipeline review-faces           # step 4: verify detections in a web UI
    face-pipeline write-xmp              # step 5
    face-pipeline status                 # summary of faces.db
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from face_pipeline.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "config_path", default="config.yaml", show_default=True,
              help="Path to config.yaml (see config.example.yaml).")
@click.pass_context
def cli(ctx: click.Context, config_path: str):
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command("inspect-catalog")
@click.pass_context
def inspect_catalog_cmd(ctx: click.Context):
    """Dump the schema of the face-related tables in the Lightroom catalog
    copy, to confirm/debug column-name detection."""
    from face_pipeline.lightroom.catalog_export import inspect_catalog

    cfg = Config.load(ctx.obj["config_path"])
    click.echo(inspect_catalog(cfg.catalog_copy_path))


@cli.command("export-faces")
@click.pass_context
def export_faces_cmd(ctx: click.Context):
    """Step 1: export person-tagged faces (name + bounding box) from the
    Lightroom catalog copy to a CSV."""
    from face_pipeline.lightroom.catalog_export import export_faces

    cfg = Config.load(ctx.obj["config_path"])
    count = export_faces(cfg.catalog_copy_path, cfg.labeled_faces_csv, cfg.images_root)
    click.echo(f"Exported {count} labeled faces to {cfg.labeled_faces_csv}")


@cli.command("export-status")
@click.pass_context
def export_status_cmd(ctx: click.Context):
    """Summary of the CSV exported in step 1: how many Lightroom-tagged faces
    were exported, total and per person. This counts the labeled export, not
    detections from run-inference (see `status` for that)."""
    import csv as csv_module
    from collections import Counter

    cfg = Config.load(ctx.obj["config_path"])
    if not cfg.labeled_faces_csv.exists():
        raise click.ClickException(
            f"{cfg.labeled_faces_csv} not found -- run `face-pipeline export-faces` first."
        )

    with cfg.labeled_faces_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    per_person = Counter(row["person_name"] for row in rows)
    click.echo(f"Labeled faces exported: {len(rows)}")
    for person, n in per_person.most_common():
        click.echo(f"  {person}: {n}")


@cli.command("verify-crops")
@click.option("-n", "--count", default=10, show_default=True, help="Number of sample crops to save.")
@click.option("-o", "--out-dir", default="data/verify_crops", show_default=True)
@click.pass_context
def verify_crops_cmd(ctx: click.Context, count: int, out_dir: str):
    """Debug helper: crops N labeled faces straight from the exported CSV
    (no detection model involved) and saves them as JPEGs, so you can
    visually confirm Lightroom's bounding boxes line up with the right
    person before trusting a full training run."""
    import csv as csv_module

    import cv2

    from face_pipeline.imaging import fractional_to_pixel_bbox, load_image_bgr

    cfg = Config.load(ctx.obj["config_path"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with cfg.labeled_faces_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    saved = 0
    for row in rows:
        if saved >= count:
            break
        path = Path(row["image_path"])
        if not path.exists():
            continue
        image_bgr = load_image_bgr(path)
        height, width = image_bgr.shape[:2]
        left, top, right, bottom = fractional_to_pixel_bbox(
            float(row["left"]), float(row["top"]), float(row["right"]), float(row["bottom"]), width, height
        )
        crop = image_bgr[max(0, int(top)):int(bottom), max(0, int(left)):int(right)]
        if crop.size == 0:
            continue
        out_path = out / f"{saved:03d}_{row['person_name']}.jpg"
        # cv2.imwrite mangles non-ASCII paths on Windows (it goes through a
        # narrow C file API, so UTF-8 bytes get reinterpreted with the ANSI
        # codepage). Encode in memory and write the bytes ourselves instead.
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            continue
        out_path.write_bytes(buf.tobytes())
        saved += 1

    click.echo(f"Saved {saved} sample crops to {out} -- open them and confirm each shows the named person's face.")


@cli.command("train-classifier")
@click.pass_context
def train_classifier_cmd(ctx: click.Context):
    """Step 2: train a person classifier on InsightFace embeddings of the
    faces exported in step 1."""
    from face_pipeline.recognition.embeddings import FaceEmbedder
    from face_pipeline.recognition.train_classifier import collect_training_examples, save, train

    cfg = Config.load(ctx.obj["config_path"])
    embedder = FaceEmbedder(
        model_name=cfg.recognition.model_name,
        ctx_id=cfg.recognition.ctx_id,
        det_size=cfg.recognition.detector_size,
        model_root=cfg.recognition.model_root,
    )
    examples = collect_training_examples(cfg.labeled_faces_csv, embedder)
    classifier = train(examples)
    save(classifier, cfg.classifier_path)
    click.echo(f"Trained classifier for {len(classifier.labels)} people: {classifier.labels}")


@cli.command("run-inference")
@click.option("--limit", type=int, default=None, help="Only process the first N images (for a quick test run).")
@click.option(
    "--restart", is_flag=True,
    help="Reprocess every image from scratch, ignoring the usual skip-if-unchanged check. "
         "Without this flag, a run that was previously stopped partway through simply "
         "continues where it left off. Reprocessing an image replaces its faces and resets "
         "their review decisions back to pending, same as today when a photo's mtime changes "
         "-- with --restart this applies to the whole library, not just changed files.",
)
@click.pass_context
def run_inference_cmd(ctx: click.Context, limit: int | None, restart: bool):
    """Step 3: detect + classify faces across the full image library, storing
    results in faces.db."""
    from face_pipeline.pipeline.run_inference import run
    from face_pipeline.recognition.embeddings import FaceEmbedder
    from face_pipeline.recognition.train_classifier import load as load_classifier

    cfg = Config.load(ctx.obj["config_path"])
    embedder = FaceEmbedder(
        model_name=cfg.recognition.model_name,
        ctx_id=cfg.recognition.ctx_id,
        det_size=cfg.recognition.detector_size,
        model_root=cfg.recognition.model_root,
    )
    classifier = load_classifier(cfg.classifier_path)
    stats = run(
        cfg.images_root, cfg.faces_db, embedder, classifier,
        min_confidence=cfg.recognition.min_confidence, limit=limit, restart=restart,
    )
    click.echo(stats)


@cli.command("review-faces")
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind the review UI to.")
@click.option("--port", default=7860, show_default=True, type=int)
@click.option("--share", is_flag=True, help="Also create a public gradio.live link (e.g. to review from another device).")
@click.pass_context
def review_faces_cmd(ctx: click.Context, host: str, port: int, share: bool):
    """Step 4: interactively verify detected faces (cropped photo + predicted
    name) in a local web UI before write-xmp writes them into Lightroom.
    Only faces you approve here get written."""
    try:
        from face_pipeline.review.ui import launch
    except ImportError as exc:
        raise click.ClickException(
            "gradio is not installed. Install it with `pip install -e \".[review]\"` "
            "(or `pip install gradio`) to use this command."
        ) from exc

    cfg = Config.load(ctx.obj["config_path"])
    click.echo(f"Starting review UI at http://{host}:{port} -- open it in a browser. Ctrl+C to stop.")
    launch(cfg, host=host, port=port, share=share)


@cli.command("write-xmp")
@click.option("--dry-run", is_flag=True, help="Print what would be written without touching any files.")
@click.option("--no-backup", is_flag=True, help="Skip exiftool's automatic backup copy (embed-format files only).")
@click.option(
    "--resume", is_flag=True,
    help="Skip images already exported by a previous, interrupted write-xmp run instead of "
         "starting over from the first approved image. Has no effect after a run completes "
         "in full -- completion always clears the resume markers, so the next run (resumed "
         "or not) processes everything again.",
)
@click.pass_context
def write_xmp_cmd(ctx: click.Context, dry_run: bool, no_backup: bool, resume: bool):
    """Step 5: write approved faces back out as MWG face regions (embedded
    XMP or .xmp sidecar, matching Lightroom's own convention per format).
    Only faces approved via `review-faces` are written."""
    from face_pipeline.lightroom.xmp_writer import write_approved_regions
    from face_pipeline.pipeline import db

    cfg = Config.load(ctx.obj["config_path"])
    with db.connect(cfg.faces_db) as conn:
        pending = db.review_counts(conn)["pending"]

    stats = write_approved_regions(cfg.faces_db, keep_backup=not no_backup, dry_run=dry_run, resume=resume)

    click.echo(f"Wrote regions for {stats['written']} images ({stats['failed']} failed).")
    if resume and stats["skipped_already_exported"]:
        click.echo(
            f"Skipped {stats['skipped_already_exported']} image(s) already exported by "
            "a previous, interrupted run."
        )
    if pending:
        click.echo(
            f"{pending} detected face(s) are still pending review and were skipped. "
            "Run `face-pipeline review-faces` to approve/reject them, then re-run write-xmp."
        )
    click.echo(
        "Next: in Lightroom, select the affected photos and run "
        "Metadata > Read Metadata from Files so the new face regions appear."
    )


@cli.command("status")
@click.pass_context
def status_cmd(ctx: click.Context):
    """Quick summary of faces.db: image/face counts, per-person counts."""
    from face_pipeline.pipeline import db

    cfg = Config.load(ctx.obj["config_path"])
    with db.connect(cfg.faces_db) as conn:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        per_person = conn.execute(
            "SELECT person_name, COUNT(*) AS n FROM faces GROUP BY person_name ORDER BY n DESC"
        ).fetchall()
        review = db.review_counts(conn)

    click.echo(f"Images processed: {n_images}")
    click.echo(f"Faces detected:   {n_faces}")
    for row in per_person:
        click.echo(f"  {row['person_name']}: {row['n']}")
    click.echo(
        f"Review status:    {review['pending']} pending, {review['approved']} approved, "
        f"{review['rejected']} rejected"
    )
    if review["pending"]:
        click.echo("  Run `face-pipeline review-faces` to review pending faces before write-xmp.")


if __name__ == "__main__":
    cli()
