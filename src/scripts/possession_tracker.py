import math


class PossessionTracker:
    def __init__(self, proximity_px=80):
        self.prox, self.counts = proximity_px, [0, 0]

    def update(self, b_box, p_boxes, team_ids):
        """Updates the possession counts based on the proximity of the ball to players of each team."""
        if not b_box:
            return -1

        # Get the center of the ball bounding box
        bx, by = (b_box[0] + b_box[2]) / 2, (b_box[1] + b_box[3]) / 2

        # Find the closest player to the ball
        distance, team_id = min(
            (
                (
                    math.hypot(
                        (bx - max(x1, min(x2, bx))), (by - max(y1, min(y2, by)))
                    ),
                    tid,
                )
                for (x1, y1, x2, y2), tid in zip(p_boxes, team_ids)
                if tid >= 0
            ),
            default=(float("inf"), -1),
        )

        # If the closest player is within proximity, increment their team's count
        if distance < self.prox:
            self.counts[team_id] += 1
            return team_id

        return -1

    def percentage(self):
        """Returns the possession percentage for each team."""
        total = sum(self.counts)
        return [c / total * 100 for c in self.counts] if total > 0 else [50.0, 50.0]
