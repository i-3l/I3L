"""Trim a shared transparent border from Figure 4 PNGs without resampling."""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--individual-bottom", action="store_true",
                        help="Trim each panel below its labels, leaving a six-pixel buffer.")
    parser.add_argument("--padding", type=int, default=6, help="Label safety buffer in pixels.")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output directory to preserve the source images.")
    paths = sorted(args.source.glob("timeline_*_bare.png"))
    if len(paths) != 8:
        parser.error("Expected eight subplot PNGs.")
    images = [Image.open(path).convert("RGBA") for path in paths]
    assert len({im.height for im in images}) == 1
    bounds = [im.getchannel("A").getbbox() for im in images]
    crop = (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds))
    args.output.mkdir(parents=True, exist_ok=True)
    panel_crops = {}
    for path, im, bound in zip(paths, images, bounds):
        panel_crop = (crop[0], crop[1], crop[2] + args.padding,
                      min(im.height, bound[3] + args.padding)) if args.individual_bottom else crop
        im.crop(panel_crop).save(args.output / path.name, dpi=im.info.get("dpi", (300, 300)))
        panel_crops[path.name] = panel_crop
    metadata = {"source": str(args.source.resolve()), "source_size": images[0].size,
                "crop_box": crop, "output_sizes": {
                    name: [box[2] - box[0], box[3] - box[1]] for name, box in panel_crops.items()},
                "resampled": False, "panel_crop_boxes": panel_crops}
    (args.output / "crop_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
