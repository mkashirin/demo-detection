import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


class TestTaskDatasetPreprocessor:
    """A preprocessor for aligning CVAT video-mode XML annotations with image
    sequences, specifically designed for panoramic BEV datasets.
    """

    def __init__(
        self,
        root: str,
        class_map: Dict[str, int] = None,
        image_dims: Tuple[float, float] = (1882.0, 862.0),
        split_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.class_map = class_map or {"car": 0, "truck": 1}
        self.width, self.height = image_dims
        self.split_ratio = split_ratio
        self.seed = seed

        self.subsets = ["train", "val"]
        self.dirs = {
            "images": self.root / "images",
            "labels": self.root / "labels",
        }

    def _setup_directories(self):
        for folder in self.dirs.values():
            for subset in self.subsets:
                (folder / subset).mkdir(parents=True, exist_ok=True)

    def _parse_cvat_xml(self, xml_path: str) -> Dict[int, List[str]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_data = {}

        for tag in root.findall("image"):
            raw_name = tag.get("name")
            try:
                frame_index = int("".join(filter(str.isdigit, raw_name)))
            except ValueError:
                continue

            labels = []
            for box in tag.findall("box"):
                label_name = box.get("label")
                if label_name in self.class_map:
                    class_id = self.class_map[label_name]
                    xtl, ytl = box.get("xtl"), box.get("ytl")
                    xtl, ytl = float(xtl), float(ytl)
                    xbr, ybr = box.get("xbr"), box.get("ybr")
                    xbr, ybr = float(xbr), float(ybr)

                    x_center = (xtl + xbr) / 2.0 / self.width
                    y_center = (ytl + ybr) / 2.0 / self.height
                    w = (xbr - xtl) / self.width
                    h = (ybr - ytl) / self.height

                    labels.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f}"
                        + f" {w:.6f} {h:.6f}"
                    )

            if labels:
                image_data[frame_index] = labels
        return image_data

    def _get_short_dest_name(self, dir_name: str, original_name: str) -> str:
        base_name = original_name.split("__")[0]
        if not base_name.lower().endswith(".jpg"):
            base_name += ".jpg"
        return f"{dir_name}_{base_name}"

    def _save_subset(self, data: List[Dict[str, Any]], subset: str):
        for item in tqdm(data, desc=f"Writing {subset}"):
            dest_image = self.dirs["images"] / subset / item["dest_name"]
            shutil.copy(item["src"], dest_image)

            label_name = Path(item["dest_name"]).with_suffix(".txt")
            dest_label = self.dirs["labels"] / subset / label_name
            with open(dest_label, "w") as f:
                f.write("\n".join(item["labels"]))

    def run(self, data_pairs: List[Tuple[str, str]]):
        self._setup_directories()
        all_samples = []

        for image_dir_str, xml_path_str in data_pairs:
            image_dir = Path(image_dir_str)
            xml_path = Path(xml_path_str)

            print(f"Processing {image_dir.name}...")
            if not xml_path.exists() or not image_dir.exists():
                print(f"Warning: Missing files for {image_dir.name}")
                continue

            sorted_files = sorted(
                [
                    f_name
                    for f_name in os.listdir(image_dir)
                    if f_name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

            annotations = self._parse_cvat_xml(str(xml_path))

            for frame_index, labels in annotations.items():
                if frame_index < len(sorted_files):
                    actual_name = sorted_files[frame_index]

                    all_samples.append(
                        {
                            "src": image_dir / actual_name,
                            "dest_name": self._get_short_dest_name(
                                image_dir.name, actual_name
                            ),
                            "labels": labels,
                        }
                    )

        if not all_samples:
            print("No valid samples found! Check CLASS_MAP or file paths")
            return

        random.seed(self.seed)
        random.shuffle(all_samples)
        split_point = int(len(all_samples) * (1 - self.split_ratio))

        train_data = all_samples[:split_point]
        val_data = all_samples[split_point:]

        self._save_subset(train_data, "train")
        self._save_subset(val_data, "val")

        print("\nProcessing complete!")
        print(
            f"Total: {len(all_samples)}"
            + f"\nTrain: {len(train_data)}"
            + f"\nVal: {len(val_data)}"
        )
