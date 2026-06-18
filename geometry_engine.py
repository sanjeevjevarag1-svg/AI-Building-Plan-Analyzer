import cv2
import numpy as np


def extract_rooms(closed):
    inverted = cv2.bitwise_not(closed)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8
    )

    rooms = []

    h, w = closed.shape

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 5000:
            continue

        # reject full image background
        if bw > 0.85 * w and bh > 0.85 * h:
            continue

        # reject tiny thin artifacts
        if bw < 50 or bh < 50:
            continue

        aspect = bw / bh

        if aspect > 8 or aspect < 0.12:
            continue

        rooms.append({
            "x": int(x),
            "y": int(y),
            "width": int(bw),
            "height": int(bh),
            "area": int(area)
        })

    return rooms