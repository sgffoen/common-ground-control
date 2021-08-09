import cv2
import math as m
import numpy as np


class LineArtist(object):
    def __init__(self):
        img = 255 * np.ones(shape=[m.floor(256),
                                   m.floor(256),
                                   3], dtype=np.uint8)
        self.original_image = img
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
        self.count = 0

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if (event == cv2.EVENT_LBUTTONDOWN) and (self.count%2 == 0):
            self.image_coordinates = [(x,y)]
            self.count += 1

        # Record ending (x,y) coordintes on left mouse bottom click
        elif (event == cv2.EVENT_LBUTTONDOWN) and (self.count%2 == 1):
            self.count += 1
            self.image_coordinates.append((x,y))
            print('Line: {}, Starting: {}, Ending: {}'.format(int(self.count/2), self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,0,255), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    draw_line_widget = LineArtist()
    while True:
        img = draw_line_widget.show_image()
        cv2.imshow('image', img)
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            filename = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/' + 'test.png'
            cv2.imwrite(filename, img)

            cv2.destroyAllWindows()
            exit(1)
