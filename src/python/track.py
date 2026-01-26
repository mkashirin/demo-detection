from ultralytics import YOLO


ROOT = "../../"


def main() -> None:
    model = YOLO(ROOT + "models/demo_detection_v4e13.pt")

    results = model.track(
        source=ROOT + "dataset/video_test/10.0s_video_test.mp4",
        imgsz=1280,
        conf=0.5,
        iou=0.4,
        show=True,
        save=True,
        tracker="botsort.yaml",
        stream=True,
    )

    unique_vehicle_ids = set()
    for result in results:
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            unique_vehicle_ids.update(ids)

    unique_vehicles = len(unique_vehicle_ids)
    print(f"Total unique vehicles detected in video: {unique_vehicles}")


if __name__ == "__main__":
    main()
