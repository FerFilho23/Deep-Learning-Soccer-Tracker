import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse, cv2, torch, time
from ultralytics import YOLO

# Set up argument parser
p = argparse.ArgumentParser()
p.add_argument(
    "--model",
    type=str,
    default="runs/detect/yolo26n_baseline_960/weights/best.pt",
    help="Path to the YOLO model file",
)
p.add_argument(
    "--source",
    type=str,
    default="0",
    help="Path to the video file or camera index",
)
p.add_argument(
    "--stream",
    action="store_true",
    help="Enable streaming mode",
),
p.add_argument("--img-size", type=int, default=960, help="Image size for inference")
p.add_argument(
    "--conf-thres", type=float, default=0.25, help="Confidence threshold for detections"
)
p.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps')",
)
args = p.parse_args()

args.device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

src = args.source.lstrip("usb") if args.source.startswith("usb") else args.source
is_img = src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))

if not is_img and not args.stream:
    args.stream = (
        input("Streaming mode is not enabled. Do you want to enable it? (y/n): ")
        .strip()
        .lower()
        == "y"
    )

os.makedirs("output", exist_ok=True)
fps = 30.0

# TODO: FIX error when opening camera. Add error handling for video capture
# if not is_img:
#     cap = cv2.VideoCapture(src)
#     fps = cap.get(cv2.CAP_PROP_FPS) or fps
#     cap.release()
#     if not cap.isOpened():
#         print(f"Error: Unable to open video source {src}")
#         exit(1)

writer, frames = None, 0

for r in YOLO(args.model).predict(
    source=src,
    imgsz=args.img_size,
    conf=args.conf_thres,
    device=args.device,
    stream=True,
    verbose=False,
):
    img = r.plot()

    if is_img:
        cv2.imwrite("output/result.jpg", img)
        print("Result saved to output/result.jpg")
        break

    if not writer:
        height, width, _ = img.shape
        writer = cv2.VideoWriter(
            "output/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    writer.write(img)
    frames += 1

    if args.stream:
        cv2.imshow("YOLO Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    elif frames % int(fps) == 0:
        print(f"Processed {frames} frames", end="\r")

if writer:
    writer.release()
    print(f"\nVideo saved to output/result.mp4")
cv2.destroyAllWindows()
