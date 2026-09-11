from pathlib import Path
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
LABELS_DIR = BASE_DIR / "data" / "labels"

CLASSES = ["without_mask", "with_mask", "mask_weared_incorrect"]


def convert_box(size, box):
    width, height = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return x_center, y_center, box_width, box_height


def convert_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.findtext("filename")
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in CLASSES:
            raise ValueError(
                f"Class '{class_name}' trong {xml_file.name} khong nam trong danh sach: {CLASSES}"
            )

        class_id = CLASSES.index(class_name)
        box = (
            int(obj.find("bndbox/xmin").text),
            int(obj.find("bndbox/ymin").text),
            int(obj.find("bndbox/xmax").text),
            int(obj.find("bndbox/ymax").text),
        )
        yolo_box = convert_box((width, height), box)
        lines.append(
            f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
        )

    label_name = Path(filename).with_suffix(".txt").name
    label_file = LABELS_DIR / label_name
    label_file.write_text("\n".join(lines), encoding="utf-8")
    return label_file


def main():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    if not xml_files:
        print(f"Chua co file .xml trong: {ANNOTATIONS_DIR}")
        print("Hay dua cac file annotation XML vao thu muc nay roi chay lai:")
        print("python make_anot.py")
        return

    for xml_file in xml_files:
        label_file = convert_xml(xml_file)
        print(f"Da tao: {label_file}")

    print(f"Hoan tat: {len(xml_files)} file label.")


if __name__ == "__main__":
    main()
