import argparse
from ultralytics import YOLO


def main(args):
    # Load params
    parser = argparse.ArgumentParser(description="YOLO26 Soccer Tracking")
    parser.add_argument(
        "--model",
        type=str,
        default="../runs/detect/yolo26n_baseline_960/weights/best.pt",
        help="Path to the YOLO model file",
    )
    parser.add_argument("--source", type=str, help="Path to the video/image file")
    args = parser.parse_args(args)

    source = (
        args.source.replace("camera", "") or "0"
        if args.source.startswith("camera")
        else args.source
    )

    # Load the YOLO model
    model = YOLO(args.model)

    # Run the YOLO model on the video/image file
    for result in model.predict(source, show=True, save=True, output="output"):
        pass


if __name__ == "__main__":
    main()
