from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import time

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

try:
    import cv2
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Thieu thu vien Python: {missing_package}\n"
        "Cai dat bang lenh: python -m pip install -r requirements.txt"
    ) from exc


DEFAULT_MODEL = "best.pt"
DEFAULT_NO_MASK_CONF = 0.5
DEFAULT_MASK_CONF = 0.75
DEFAULT_MAX_MASK_AREA = 0.0
DEFAULT_MAX_MASK_HEIGHT = 0.0
DEFAULT_MAX_SKIN_RATIO_FOR_MASK = 0.35
DEFAULT_CONFIRM_SECONDS = 3.0
DEFAULT_MASK_GRACE_SECONDS = 1.5
DEFAULT_ALERT_SECONDS = 0.8
DEFAULT_ALERT_COOLDOWN = 2.0
DEFAULT_SNAPSHOT_DIR = "human_check"

ALERT_STATES = {"no_mask", "no_mask_visible_face", "uncertain", "uncertain_cover", "no_face"}


def load_model(model_path):
    model_file = Path(model_path)
    if model_path == DEFAULT_MODEL and not model_file.exists():
        trained_models = sorted(
            Path("runs").glob("detect/train*/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if trained_models:
            model_file = trained_models[0]

    if not model_file.exists():
        raise FileNotFoundError(
            f"Khong tim thay file model: {model_file.resolve()}\n"
            "Hay dat file best.pt vao thu muc project hoac chay voi --model <duong_dan_model>."
        )
    return YOLO(str(model_file))


def get_lower_face_skin_ratio(image, box):
    x1, y1, x2, y2 = box
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))

    box_height = y2 - y1
    if x2 <= x1 or box_height <= 0:
        return 0.0

    lower_y1 = y1 + int(box_height * 0.45)
    lower_face = image[lower_y1:y2, x1:x2]
    if lower_face.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return cv2.countNonZero(skin_mask) / skin_mask.size


def get_display_prediction(
    name,
    confidence,
    box,
    image,
    no_mask_conf,
    mask_conf,
    max_mask_area,
    max_mask_height,
    max_skin_ratio_for_mask,
):
    is_mask = name in {"with_mask", "mask"}
    if is_mask:
        if confidence < mask_conf:
            return "uncertain", (0, 255, 255)

        x1, y1, x2, y2 = box
        image_shape = image.shape
        image_height, image_width = image_shape[:2]
        box_area_ratio = ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
        box_height_ratio = (y2 - y1) / image_height

        area_too_large = max_mask_area > 0 and box_area_ratio > max_mask_area
        height_too_large = max_mask_height > 0 and box_height_ratio > max_mask_height
        if area_too_large or height_too_large:
            return "uncertain_cover", (0, 255, 255)

        skin_ratio = get_lower_face_skin_ratio(image, box)
        if skin_ratio > max_skin_ratio_for_mask:
            return "no_mask_visible_face", (0, 0, 255)

        return "mask", (0, 255, 0)

    if confidence >= no_mask_conf:
        return "no_mask", (0, 0, 255)
    return None, None


def get_frame_status(predictions):
    if not predictions:
        return "no_face"
    if "no_mask" in predictions:
        return "no_mask"
    if "uncertain_cover" in predictions:
        return "uncertain_cover"
    if "uncertain" in predictions:
        return "uncertain"
    if "mask" in predictions:
        return "mask"
    return "no_face"


def draw_status_panel(image, monitor_status, frame_status, mask_elapsed, confirm_seconds):
    if monitor_status == "monitoring":
        if frame_status == "mask":
            text = "MONITORING: MASK OK"
            color = (0, 160, 0)
        elif frame_status == "mask_tracking":
            text = "MONITORING: MASK TRACKING"
            color = (0, 160, 0)
        else:
            text = f"ALERT: {frame_status}"
            color = (0, 0, 255)
    else:
        text = f"CHECKING MASK: {mask_elapsed:.1f}/{confirm_seconds:.1f}s"
        color = (0, 180, 255)

    cv2.rectangle(image, (10, 10), (420, 48), color, -1)
    cv2.putText(
        image,
        text,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_alert_snapshot(frame, frame_status, snapshot_dir, monitor_state, now):
    if now - monitor_state["last_alert_at"] < monitor_state["alert_cooldown"]:
        return

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = snapshot_dir / f"{timestamp}_{frame_status}.jpg"
    cv2.imwrite(str(snapshot_path), frame)
    monitor_state["last_alert_at"] = now
    print(f"Da luu frame can human check: {snapshot_path.resolve()}")


def update_monitor_state(
    frame_status,
    monitor_state,
    confirm_seconds,
    mask_grace_seconds,
    alert_seconds,
    now,
):
    effective_status = frame_status

    if frame_status == "mask":
        if monitor_state["mask_started_at"] is None:
            monitor_state["mask_started_at"] = now
        mask_elapsed = now - monitor_state["mask_started_at"]
        monitor_state["last_mask_at"] = now
        monitor_state["alert_started_at"] = None
        alert_elapsed = 0.0
    else:
        monitor_state["mask_started_at"] = None
        mask_elapsed = 0.0
        recently_saw_mask = (
            monitor_state["last_mask_at"] is not None
            and now - monitor_state["last_mask_at"] <= mask_grace_seconds
        )
        can_use_grace = frame_status in {"uncertain", "uncertain_cover", "no_face"}

        if monitor_state["accepted"] and recently_saw_mask and can_use_grace:
            effective_status = "mask_tracking"
            monitor_state["alert_started_at"] = None
            alert_elapsed = 0.0
        elif frame_status in {"no_mask", "no_mask_visible_face"}:
            monitor_state["last_mask_at"] = None
            if monitor_state["accepted"] and frame_status in ALERT_STATES:
                if monitor_state["alert_started_at"] is None:
                    monitor_state["alert_started_at"] = now
                alert_elapsed = now - monitor_state["alert_started_at"]
            else:
                monitor_state["alert_started_at"] = None
                alert_elapsed = 0.0
        elif monitor_state["accepted"] and frame_status in ALERT_STATES:
            if monitor_state["alert_started_at"] is None:
                monitor_state["alert_started_at"] = now
            alert_elapsed = now - monitor_state["alert_started_at"]
        else:
            monitor_state["alert_started_at"] = None
            alert_elapsed = 0.0

    if not monitor_state["accepted"] and mask_elapsed >= confirm_seconds:
        monitor_state["accepted"] = True
        print("Da xac nhan deo khau trang on dinh, bat dau monitoring.")

    monitor_status = "monitoring" if monitor_state["accepted"] else "checking"
    alert_ready = monitor_status == "monitoring" and alert_elapsed >= alert_seconds
    return monitor_status, effective_status, mask_elapsed, alert_elapsed, alert_ready


def analyze_frame(
    results,
    image,
    classes,
    no_mask_conf,
    mask_conf,
    max_mask_area,
    max_mask_height,
    max_skin_ratio_for_mask,
):
    boxes = results[0].boxes.xyxy.tolist()
    labels = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    predictions = []

    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        name = classes[int(label)]
        prediction, color = get_display_prediction(
            name,
            conf,
            (x1, y1, x2, y2),
            image,
            no_mask_conf,
            mask_conf,
            max_mask_area,
            max_mask_height,
            max_skin_ratio_for_mask,
        )
        if prediction is None:
            continue

        predictions.append(prediction)
        text = f"{prediction} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image, get_frame_status(predictions)


def run_image(
    model,
    image_path,
    no_mask_conf,
    mask_conf,
    max_mask_area,
    max_mask_height,
    max_skin_ratio_for_mask,
    output_path=None,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {Path(image_path).resolve()}")

    results = model(image)
    image, frame_status = analyze_frame(
        results,
        image,
        model.names,
        no_mask_conf,
        mask_conf,
        max_mask_area,
        max_mask_height,
        max_skin_ratio_for_mask,
    )
    print(f"Trang thai anh: {frame_status}")

    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Da luu ket qua vao: {Path(output_path).resolve()}")
    else:
        cv2.imshow("YOLOv8 mask detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(
    model,
    source,
    no_mask_conf,
    mask_conf,
    max_mask_area,
    max_mask_height,
    max_skin_ratio_for_mask,
    confirm_seconds,
    mask_grace_seconds,
    alert_seconds,
    snapshot_dir,
    alert_cooldown,
    output_path=None,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video/webcam: {source}")

    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        codec = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    cv2.namedWindow("YOLOv8 mask detection", cv2.WINDOW_NORMAL)
    monitor_state = {
        "accepted": False,
        "mask_started_at": None,
        "last_mask_at": None,
        "alert_started_at": None,
        "last_alert_at": 0.0,
        "alert_cooldown": alert_cooldown,
    }
    snapshot_dir = Path(snapshot_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.monotonic()
        results = model(frame)
        frame, frame_status = analyze_frame(
            results,
            frame,
            model.names,
            no_mask_conf,
            mask_conf,
            max_mask_area,
            max_mask_height,
            max_skin_ratio_for_mask,
        )
        monitor_status, effective_status, mask_elapsed, alert_elapsed, alert_ready = update_monitor_state(
            frame_status,
            monitor_state,
            confirm_seconds,
            mask_grace_seconds,
            alert_seconds,
            now,
        )
        draw_status_panel(frame, monitor_status, effective_status, mask_elapsed, confirm_seconds)

        if alert_ready:
            save_alert_snapshot(frame, effective_status, snapshot_dir, monitor_state, now)

        cv2.imshow("YOLOv8 mask detection", frame)

        if out:
            out.write(frame)

        if cv2.waitKey(24) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser(description="Chay nhan dien deo khau trang bang YOLOv8.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Duong dan file .pt.")
    parser.add_argument("--image", help="Duong dan anh dau vao.")
    parser.add_argument("--video", help="Duong dan video dau vao.")
    parser.add_argument("--webcam", type=int, help="Index webcam, thuong la 0.")
    parser.add_argument("--output", help="Duong dan file ket qua anh/video.")
    parser.add_argument(
        "--no-mask-conf",
        type=float,
        default=DEFAULT_NO_MASK_CONF,
        help="Nguong tin cay de bao no_mask, mac dinh 0.5.",
    )
    parser.add_argument(
        "--mask-conf",
        type=float,
        default=DEFAULT_MASK_CONF,
        help="Nguong tin cay de xac nhan mask, mac dinh 0.75.",
    )
    parser.add_argument(
        "--max-mask-area",
        type=float,
        default=DEFAULT_MAX_MASK_AREA,
        help="Ti le dien tich lon nhat cua box mask so voi frame; 0 la tat check nay.",
    )
    parser.add_argument(
        "--max-mask-height",
        type=float,
        default=DEFAULT_MAX_MASK_HEIGHT,
        help="Ti le chieu cao lon nhat cua box mask so voi frame; 0 la tat check nay.",
    )
    parser.add_argument(
        "--max-skin-ratio-for-mask",
        type=float,
        default=DEFAULT_MAX_SKIN_RATIO_FOR_MASK,
        help="Ti le mau da toi da o nua duoi mat de van accept mask, mac dinh 0.35.",
    )
    parser.add_argument(
        "--confirm-seconds",
        type=float,
        default=DEFAULT_CONFIRM_SECONDS,
        help="So giay phai thay mask lien tuc truoc khi accept, mac dinh 3.0.",
    )
    parser.add_argument(
        "--mask-grace-seconds",
        type=float,
        default=DEFAULT_MASK_GRACE_SECONDS,
        help="So giay giu trang thai mask khi bi nghieng/rung lam mat detect tam thoi, mac dinh 1.5.",
    )
    parser.add_argument(
        "--alert-seconds",
        type=float,
        default=DEFAULT_ALERT_SECONDS,
        help="So giay do/vang lien tuc moi luu human_check, mac dinh 0.8.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=DEFAULT_SNAPSHOT_DIR,
        help="Thu muc luu frame do/vang de human check, mac dinh human_check.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=DEFAULT_ALERT_COOLDOWN,
        help="Khoang cach giua 2 lan luu frame can check, mac dinh 2.0 giay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_sources = [args.image is not None, args.video is not None, args.webcam is not None]
    if sum(selected_sources) == 0 and Path("image.png").exists():
        args.image = "image.png"
        args.output = args.output or "output_test.jpg"
        print("Khong thay nguon dau vao, tu dong dung: --image image.png --output output_test.jpg")
    elif sum(selected_sources) != 1:
        raise ValueError("Hay chon dung 1 nguon: --image, --video hoac --webcam.")

    model = load_model(args.model)

    if args.image:
        run_image(
            model,
            args.image,
            args.no_mask_conf,
            args.mask_conf,
            args.max_mask_area,
            args.max_mask_height,
            args.max_skin_ratio_for_mask,
            args.output,
        )
    elif args.video:
        run_video(
            model,
            args.video,
            args.no_mask_conf,
            args.mask_conf,
            args.max_mask_area,
            args.max_mask_height,
            args.max_skin_ratio_for_mask,
            args.confirm_seconds,
            args.mask_grace_seconds,
            args.alert_seconds,
            args.snapshot_dir,
            args.alert_cooldown,
            args.output,
        )
    else:
        run_video(
            model,
            args.webcam,
            args.no_mask_conf,
            args.mask_conf,
            args.max_mask_area,
            args.max_mask_height,
            args.max_skin_ratio_for_mask,
            args.confirm_seconds,
            args.mask_grace_seconds,
            args.alert_seconds,
            args.snapshot_dir,
            args.alert_cooldown,
            args.output,
        )


if __name__ == "__main__":
    main()
