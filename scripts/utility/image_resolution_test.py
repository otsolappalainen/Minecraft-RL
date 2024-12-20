import cv2
import numpy as np
from PIL import ImageGrab
import tkinter as tk

class ScreenshotEditor:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        
        # Capture initial screenshot
        screenshot = ImageGrab.grab(bbox=(0, 0, 1200, 1200))
        self.image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Create window and trackbars
        cv2.namedWindow('Screenshot Editor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Screenshot Editor', 800, 800)
        
        cv2.createTrackbar('X', 'Screenshot Editor', 0, 1200, self.on_change)
        cv2.createTrackbar('Y', 'Screenshot Editor', 0, 1200, self.on_change)
        cv2.createTrackbar('Width', 'Screenshot Editor', 100, 1200, self.on_change)
        cv2.createTrackbar('Height', 'Screenshot Editor', 100, 1200, self.on_change)

    def on_change(self, value):
        self.x = cv2.getTrackbarPos('X', 'Screenshot Editor')
        self.y = cv2.getTrackbarPos('Y', 'Screenshot Editor')
        self.width = cv2.getTrackbarPos('Width', 'Screenshot Editor')
        self.height = cv2.getTrackbarPos('Height', 'Screenshot Editor')
        self.update_display()

    def update_display(self):
        # Create copy of original image
        display = self.image.copy()
        
        # Draw green rectangle
        cv2.rectangle(
            display, 
            (self.x, self.y), 
            (self.x + self.width, self.y + self.height),
            (0, 255, 0),  # Green in BGR
            2
        )
        
        cv2.imshow('Screenshot Editor', display)

    def run(self):
        self.update_display()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    editor = ScreenshotEditor()
    editor.run()