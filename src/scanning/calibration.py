import cv2
import numpy as np
if __name__=="__main__":
    from scan import scan
else:
    from .scan import scan


__IMG__ = scan(type='rgb').astype(np.uint8)
__CORNER_PTS__ = []


def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(__CORNER_PTS__) < 4:
            __CORNER_PTS__.append((x, y))
        else:
            print('all corner points are already saved')

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(__IMG__, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', __IMG__)


def skew_correction():
    pass


def img_calibration():

    # displaying the image
    cv2.imshow('image', __IMG__)

    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    try:
        print("TOP LEFT: {}\nTOP RIGHT: {}\nBOTTOM LEFT: {}\nBOTTOM RIGHT: {}". format(__CORNER_PTS__[0],
                                                                                __CORNER_PTS__[1],
                                                                                __CORNER_PTS__[2],
                                                                                __CORNER_PTS__[3],))
        skew_correction()
    except:
        print("ATTENTION: Calibration unsuccesful. 4 points are required")


if __name__=="__main__":

    img_calibration()
