from dataclasses import dataclass
from pathlib import Path
from lxml import etree

@dataclass(frozen=True)
class Sample:
    image_name: str
    image_path: Path
    gt_text: str
    gt_bbox: tuple[int, int, int, int]  # x1,y1,x2,y2

def load_samples(annotations_path: Path, photos_dir: Path) -> list[Sample]:
    tree = etree.parse(str(annotations_path))
    root = tree.getroot()

    samples: list[Sample] = []
    for img in root.findall("image"):
        name = img.get("name")
        box = img.find("box")
        if box is None:
            continue

        xtl = int(float(box.get("xtl")))
        ytl = int(float(box.get("ytl")))
        xbr = int(float(box.get("xbr")))
        ybr = int(float(box.get("ybr")))

        attr = box.find("attribute")
        gt = (attr.text or "").strip().upper() if attr is not None else ""

        samples.append(
            Sample(
                image_name=name,
                image_path=photos_dir / name,
                gt_text=gt,
                gt_bbox=(xtl, ytl, xbr, ybr),
            )
        )
    return samples
