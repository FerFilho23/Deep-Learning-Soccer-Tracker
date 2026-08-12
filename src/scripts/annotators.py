import cv2
import numpy as np

DRAW_REF, DRAW_BALL = True, True


def draw_annotations(image, boxes, cls_ids, names, team_ids, team_colors, scale=2):
    for i, (box, cid) in enumerate(zip(boxes, cls_ids)):
        x1, y1, x2, y2 = map(int, box)
        lbl, xc, yc = names.get(int(cid), "").lower(), (x1 + x2) // 2, (y1 + y2) // 2

        if "ball" in lbl:
            if DRAW_BALL:
                cv2.fillPoly(
                    image,
                    [np.array([[xc, yc - 10], [xc - 14, yc - 30], [xc + 14, yc - 30]])],
                    color=(0, 255, 255),
                )
                cv2.circle(image, (xc, yc), 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            continue

        if "referee" in lbl:
            if DRAW_REF:
                ax, ay = min(max(14, (x2 - x1) // 2 + 10), 45), min(
                    max(7, (x2 - x1) // 6 + 3), 20
                )

                cv2.ellipse(
                    image,
                    (xc, y2),
                    (ax + 5, ay + 3),
                    0,
                    -45,
                    235,
                    (0, 140, 140),
                    7,
                    lineType=cv2.LINE_AA,
                )

                cv2.ellipse(
                    image,
                    (xc, y2),
                    (ax, ay),
                    0,
                    -45,
                    235,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
        elif (tid := team_ids[i] if i < len(team_ids) else -1) >= 0:
            pts = np.array(
                [
                    [xc, y1 - 5 * scale],
                    [xc - 10 * scale, y1 - 20 * scale],
                    [xc + 10 * scale, y1 - 20 * scale],
                ],
            )
            cv2.fillPoly(image, [pts], team_colors[tid])


def jersey_color(img, x1, y1, x2, y2):
    if not (crop := img[y1 : (y1 + y2) // 2, x1:x2]).size:
        return None

    mask = (
        cv2.inRange(
            cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (35, 40, 40), (85, 255, 255)
        ).ravel()
        == 0
    )
    ng = crop.reshape(-1, 3)[mask]
    return tuple(map(int, ng.mean(axis=0))) if len(ng) >= 20 else None
