"""Human-readable input selections, recorded before launching method processes."""
from __future__ import annotations

from pathlib import Path

from .training_inputs import selection_frames


def _selection(label: str, selection: dict, resolved: dict | None = None) -> str:
    cameras = (resolved or {}).get("camera_ids", selection.get("camera_ids"))
    frames = selection_frames(selection)
    camera_text = str(cameras) if cameras is not None else f"first {selection['views']} cameras"
    step = selection.get("frame_range", {}).get("step", 1)
    return (f"  {label}: views={len(cameras) if cameras is not None else selection['views']}, "
            f"timesteps={len(frames)}, cameras={camera_text}, "
            f"frames={frames[0]}..{frames[-1]} (step={step})")


def _image_sizes(subject: dict, selections: list[tuple[str, dict]]) -> list[str]:
    """Probe one selected image per camera/split; do not decode whole sequences."""
    if subject.get("dataset") != "thuman":
        return ["  Image dimensions: unavailable before method decoding for this dataset"]
    try:
        from PIL import Image
        root = Path(subject["images"])
        cameras = {int(p.name[3:]): p for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("cam") and p.name[3:].isdigit()}
        lines = []
        for label, selection in selections:
            ids = selection.get("camera_ids", sorted(cameras)[:selection.get("views", 0)])
            frames = set(selection_frames(selection))
            sizes = set()
            missing = []
            for camera in ids:
                folder = cameras.get(camera)
                sample = next((p for p in sorted(folder.iterdir())
                               if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                               and p.stem.isdigit() and int(p.stem) in frames), None) if folder else None
                if sample is None:
                    missing.append(camera)
                    continue
                with Image.open(sample) as image:
                    sizes.add(image.size)
            rendered = ", ".join(f"{w}x{h}" for w, h in sorted(sizes)) or "unavailable"
            lines.append(f"  {label} source images (width x height; one sample per camera): {rendered}")
            resolution = subject["training_config"]["resolution"]
            if sizes:
                effective = set()
                for width, height in sizes:
                    scale = min(resolution / max(width, height), 1.0) if resolution else 1.0
                    effective.add((int(width * scale), int(height * scale)))
                rendered_effective = ", ".join(f"{w}x{h}" for w, h in sorted(effective))
                lines.append(f"  {label} image size after configured resize (width x height, from samples): {rendered_effective}")
            if missing:
                lines.append(f"  {label} cameras without a selected image: {missing}")
        return lines
    except (ImportError, OSError, ValueError, KeyError) as exc:
        return [f"  Image dimensions unavailable: {exc}"]


def build_run_summary(method: str, dataset, extra_args: list[str], devices: tuple[str, ...]) -> str:
    lines = [f"Run inputs: method={method}, dataset={dataset.kind}",
             f"Dataset path: {dataset.root}",
             f"GPU pool: {','.join(devices) if devices else 'environment/default'}"]
    for subject in dataset.subjects:
        lines.append(f"Subject: {subject['subject_id']}")
        train = subject.get("training_config")
        if not train:
            lines.append("  Train/eval selection: method defaults or command arguments (no shared run config)")
            continue
        lines.append(_selection("train", train, subject.get("training_selection")))
        resolution = train["resolution"]
        lines.append(f"  Configured resolution: {resolution}" + (" (source image size)" if resolution == 0 else " (maximum image edge in pixels)"))
        evaluation = subject.get("evaluation_config", {})
        for name, selection in evaluation.items():
            lines.append(_selection(f"eval/{name}", selection))
        if not evaluation:
            lines.append("  Eval: no explicit split configured; method/evaluation defaults apply")
        lines.extend(_image_sizes(subject, [("train", train), *[(f"eval/{k}", v) for k, v in evaluation.items()]]))
    if extra_args:
        lines.append(f"Method arguments: {extra_args}")
    return "\n".join(lines) + "\n"
