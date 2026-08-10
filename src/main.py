import argparse
from ultralytics import YOLO


def main():
    # Load params
    parser = argparse.ArgumentParser(description="YOLO26 Soccer Tracking")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/yolo26n_baseline_960/weights/best.pt",
        help="Path to the YOLO model file",
    )
    parser.add_argument("--source", type=str, help="Path to the video/image file")
    args = parser.parse_args()

    source = (
        args.source.replace("usb", "") or "0"
        if args.source.startswith("usb")
        else args.source
    )

    # Load the YOLO model
    model = YOLO(args.model)

    # Run the YOLO model on the video/image file
    for result in model.predict(
        source=source, show=True, stream=True, save=True, save_dir="output"
    ):
        pass


if __name__ == "__main__":
    main()
