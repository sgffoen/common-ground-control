import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2


def live_scan_stream():

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)
    else:
        print("Kinect device found!")

        import ktb

        k = ktb.Kinect()
        while True:
            # Specify as many types as you want here
            color_frame = k.get_frame(ktb.COLOR)
            color_frame = np.flipud(color_frame)

            cv2.imshow('frame', color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    live_scan_stream()
