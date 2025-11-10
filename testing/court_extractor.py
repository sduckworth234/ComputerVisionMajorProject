import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from utils.data_loader import DataLoader

import cv2
import numpy as np

class CourtExtractor:
    def __init__(self, debug=False):
        self.debug = debug

    def get_court_corners(self, img_path):
        # --- Load and preprocess ---
        frame = cv2.imread(img_path)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Gaussian blur
        blur = cv2.GaussianBlur(L, (5,5), 0)

        # Adaptive threshold to isolate bright lines
        thresh = cv2.adaptiveThreshold(blur, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, -5)
        
        mask_roi = np.zeros_like(thresh)
        mask_roi[150:700, 100:1200] = 255
        thresh = cv2.bitwise_and(thresh, mask_roi)


        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded Image")
        plt.show()
        # Edge detection
        edges = cv2.Canny(closed, 50, 150)

        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")
        plt.show()

        # --- Find contours ---
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Pick largest rectangular contour
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best = approx
                best_area = area

        if best is None:
            raise RuntimeError("No suitable 4-point court contour found")

        # --- Order the four points ---
        pts = best.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],  # top-left
            pts[np.argmin(diff)],  # top-right
            pts[np.argmax(diff)],  # bottom-left
            pts[np.argmax(s)]   # bottom-right
        ], dtype=np.float32)

        if self.debug:
            vis = frame.copy()
            cv2.drawContours(vis, [best], -1, (0,255,0), 3)
            for i, (x,y) in enumerate(ordered.astype(int)):
                cv2.circle(vis, (x,y), 8, (0,0,255), -1)
                cv2.putText(vis, f"{i}", (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Court Perimeter", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ordered
