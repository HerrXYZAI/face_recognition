"""Step 3.5 (recommended, run between run-inference and write-xmp): a local
Gradio web UI for reviewing detected faces before they're written into
Lightroom.

Each candidate face is shown as a padded crop next to its predicted name and
confidence. A human approves it (optionally correcting the name first),
rejects it, or skips it for later. write-xmp only writes faces with
review_status='approved' (see face_pipeline.pipeline.db.iter_faces_for_xmp),
so nothing reaches Lightroom without a human having looked at it.
"""
from __future__ import annotations

import datetime
import logging
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from face_pipeline.config import Config
from face_pipeline.imaging import load_image_bgr
from face_pipeline.pipeline import db

logger = logging.getLogger(__name__)

# Extra margin around the tight bounding box, so the reviewer sees enough
# context (hair, ears, a bit of the scene) to actually judge the match --
# a box cropped exactly to the detector's box is often too tight to tell.
PAD_FRAC = 0.3
UNKNOWN = db.UNKNOWN_NAME


@lru_cache(maxsize=8)
def _load_cached_bgr(path: str) -> np.ndarray:
    """Small cache so reviewing several faces from the same photo (common --
    group shots) doesn't reload/redecode the full-resolution image each time."""
    return load_image_bgr(path)


def _crop_face(image_bgr: np.ndarray, left: float, top: float, right: float, bottom: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    box_w, box_h = right - left, bottom - top
    pad_x, pad_y = box_w * PAD_FRAC, box_h * PAD_FRAC
    x1, y1 = max(0, int(left - pad_x)), max(0, int(top - pad_y))
    x2, y2 = min(width, int(right + pad_x)), min(height, int(bottom + pad_y))
    crop = image_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


def _counts_markdown(conn) -> str:
    counts = db.review_counts(conn)
    total = sum(counts.values())
    return (
        f"**{counts.get('pending', 0)}** pending &nbsp;·&nbsp; "
        f"**{counts.get('approved', 0)}** approved &nbsp;·&nbsp; "
        f"**{counts.get('rejected', 0)}** rejected &nbsp;·&nbsp; {total} faces total"
    )


def _render(queue: list[dict], idx: int):
    """Advances from idx until it finds a face whose image loads (skipping
    over moved/deleted files rather than getting stuck), or runs out.
    Returns (crop_or_None, info_markdown, idx, face_or_None)."""
    while idx < len(queue):
        face = queue[idx]
        try:
            image_bgr = _load_cached_bgr(face["path"])
            crop = _crop_face(image_bgr, face["left"], face["top"], face["right"], face["bottom"])
        except Exception:
            logger.warning("Skipping unreadable image in review queue: %s", face["path"], exc_info=True)
            idx += 1
            continue
        info = (
            f"**{idx + 1} / {len(queue)}**  \n"
            f"Predicted: **{face['person_name']}** "
            f"(confidence {face['confidence']:.2f}, detector {face['detector_score']:.2f})  \n"
            f"`{face['path']}`"
        )
        return crop, info, idx, face
    return None, "Queue empty -- nothing left to review with the current filters.", idx, None


def _build_gallery(queue: list[dict]) -> list[tuple[np.ndarray, str]]:
    """Renders every face in the (already-filtered) queue as a padded crop
    for the grid view. Skips faces whose source image fails to load, same as
    the one-by-one reviewer."""
    items = []
    for face in queue:
        try:
            image_bgr = _load_cached_bgr(face["path"])
            crop = _crop_face(image_bgr, face["left"], face["top"], face["right"], face["bottom"])
        except Exception:
            logger.warning("Skipping unreadable image in grid view: %s", face["path"], exc_info=True)
            continue
        caption = f"{face['person_name']} ({face['confidence']:.2f}) -- {Path(face['path']).name}"
        items.append((crop, caption))
    return items


def _name_dropdown_update(face: Optional[dict], names: list[str]):
    value = face["person_name"] if face and face["person_name"] in names else None
    return gr.update(choices=names, value=value)


def build_app(cfg: Config) -> gr.Blocks:
    def load_queue(status_filter: str, person_filter: str, min_conf: float, include_unknown: bool):
        with db.connect(cfg.faces_db) as conn:
            name_filter = None if person_filter in (None, "(any)") else person_filter
            rows = db.iter_faces_for_review(
                conn, review_status=status_filter, min_confidence=min_conf, person_name=name_filter,
            )
            if not include_unknown:
                rows = [row for row in rows if row["person_name"] != UNKNOWN]
            queue = _rows_to_dicts(rows)
            counts_md = _counts_markdown(conn)
            names = db.distinct_person_names(conn)

        image, info, idx, face = _render(queue, 0)
        person_choices = ["(any)"] + names + ([UNKNOWN] if include_unknown else [])
        person_value = person_filter if person_filter in person_choices else "(any)"
        return (
            queue, idx, names, None,
            image, info, counts_md,
            _name_dropdown_update(face, names),
            gr.update(choices=person_choices, value=person_value),
            [],  # clear any stale grid view -- it no longer matches the new queue until rebuilt
        )

    def act(
        queue: list[dict], idx: int, names: list[str], chosen_name: Optional[str],
        last_action: Optional[dict], action: str,
    ):
        if not queue or idx >= len(queue):
            return queue, idx, None, "Nothing queued -- click **Load / refresh queue**.", "", gr.update(), last_action

        face = queue[idx]
        with db.connect(cfg.faces_db) as conn:
            if action == "approve" and not chosen_name:
                counts_md = _counts_markdown(conn)
                image, info, _, cur_face = _render(queue, idx)
                warning = "**Pick a name before approving** (or click Reject to discard this face).\n\n"
                return queue, idx, image, warning + info, counts_md, _name_dropdown_update(cur_face, names), last_action

            prev = None
            if action in ("approve", "reject"):
                prev = db.get_review_state(conn, face["id"])
                if action == "approve":
                    db.set_review_status(conn, face["id"], "approved", person_name=chosen_name, reviewed_at=_now())
                else:
                    db.set_review_status(conn, face["id"], "rejected", reviewed_at=_now())
            # action == "skip": leave review_status untouched, just advance.
            counts_md = _counts_markdown(conn)

        new_last_action = {
            "face_id": face["id"], "idx": idx,
            "prev_status": prev["review_status"] if prev else None,
            "prev_person_name": prev["person_name"] if prev else None,
            "prev_reviewed_at": prev["reviewed_at"] if prev else None,
        }
        image, info, new_idx, next_face = _render(queue, idx + 1)
        return queue, new_idx, image, info, counts_md, _name_dropdown_update(next_face, names), new_last_action

    def undo_last(queue: list[dict], idx: int, names: list[str], last_action: Optional[dict]):
        with db.connect(cfg.faces_db) as conn:
            if not last_action:
                counts_md = _counts_markdown(conn)
                image, info, cur_idx, cur_face = _render(queue, idx)
                return (
                    queue, idx, image, "**Nothing to undo.**\n\n" + info, counts_md,
                    _name_dropdown_update(cur_face, names), None,
                )

            # prev_status is None for an undone "skip" (no DB write to revert).
            if last_action["prev_status"] is not None:
                db.set_review_status(
                    conn, last_action["face_id"], last_action["prev_status"],
                    person_name=last_action["prev_person_name"], reviewed_at=last_action["prev_reviewed_at"],
                )
            counts_md = _counts_markdown(conn)

        image, info, new_idx, face = _render(queue, last_action["idx"])
        return (
            queue, new_idx, image, "**Undid last action.**\n\n" + info, counts_md,
            _name_dropdown_update(face, names), None,
        )

    def build_grid(queue: list[dict]):
        return _build_gallery(queue)

    with gr.Blocks(title="face_pipeline -- review faces") as demo:
        gr.Markdown(
            "# Review detected faces\n"
            "Confirm, relabel, or reject each detected face before `write-xmp` "
            "writes it into Lightroom. **Only approved faces get written.**"
        )
        with gr.Row():
            status_filter = gr.Dropdown(
                choices=["pending", "approved", "rejected"], value="pending", label="Review status to show",
            )
            person_filter = gr.Dropdown(choices=["(any)"], value="(any)", label="Predicted person filter")
            min_conf = gr.Slider(0.0, 1.0, value=cfg.xmp_min_confidence, step=0.01, label="Min confidence to include")
            include_unknown = gr.Checkbox(value=False, label="Include 'unknown' (unclassified) faces")
        load_btn = gr.Button("Load / refresh queue")
        counts_md = gr.Markdown()

        with gr.Row():
            image = gr.Image(label="Face (padded crop)", type="numpy", height=360, interactive=False)
            info_md = gr.Markdown()

        with gr.Row():
            name_dropdown = gr.Dropdown(
                choices=[], label="Assign / confirm person name",
                allow_custom_value=True, info="Change this to correct a misidentified or 'unknown' face.",
            )
        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary")
            reject_btn = gr.Button("❌ Reject")
            skip_btn = gr.Button("⏭️ Skip (leave pending)")
            undo_btn = gr.Button("↩️ Undo last")

        with gr.Accordion("Grid view -- all filtered faces at once", open=False) as grid_accordion:
            grid_btn = gr.Button("Build / refresh grid from current filters")
            gallery = gr.Gallery(
                label="Click a thumbnail to enlarge", columns=8, height="auto",
                object_fit="contain", allow_preview=True,
            )

        queue_state = gr.State([])
        idx_state = gr.State(0)
        names_state = gr.State([])
        last_action_state = gr.State(None)

        load_inputs = [status_filter, person_filter, min_conf, include_unknown]
        load_outputs = [
            queue_state, idx_state, names_state, last_action_state,
            image, info_md, counts_md, name_dropdown, person_filter, gallery,
        ]
        act_inputs = [queue_state, idx_state, names_state, name_dropdown, last_action_state]
        act_outputs = [queue_state, idx_state, image, info_md, counts_md, name_dropdown, last_action_state]

        load_btn.click(load_queue, inputs=load_inputs, outputs=load_outputs)
        demo.load(load_queue, inputs=load_inputs, outputs=load_outputs)
        approve_btn.click(partial(act, action="approve"), inputs=act_inputs, outputs=act_outputs)
        reject_btn.click(partial(act, action="reject"), inputs=act_inputs, outputs=act_outputs)
        skip_btn.click(partial(act, action="skip"), inputs=act_inputs, outputs=act_outputs)
        undo_inputs = [queue_state, idx_state, names_state, last_action_state]
        undo_btn.click(undo_last, inputs=undo_inputs, outputs=act_outputs)
        grid_btn.click(build_grid, inputs=[queue_state], outputs=[gallery])

    return demo


def launch(cfg: Config, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    app = build_app(cfg)
    app.queue().launch(server_name=host, server_port=port, share=share)
