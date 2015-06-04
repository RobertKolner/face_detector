#!/usr/bin/env python
import argparse
import cv2
import logging
import os
import sys
import time


cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
logger = logging.getLogger(__name__)
time_of_last_image = 0

def get_cascades(directory='cascades'):
    cascades = []
    for root, dirs, files in os.walk(directory):
        cascades += filter(lambda f: f.endswith('.xml'), files)
    return cascades


def save_frame(frame, face):
    if not face:
        return

    global time_of_last_image

    now = time.time()
    if now - time_of_last_image > 3.0:
        x, y, w, h = face
        cv2.imwrite('last_face.png', frame[y:y+h, x:x+w])
        time_of_last_image = now


def show(frame, face, rect_scale=1.0, save_images=False):
    if face is not None:
        x, y, w, h = face

        margin_w = (w * rect_scale - w) / 2
        margin_h = (h * rect_scale - h)

        offset_w = margin_w / 2
        offset_h = margin_h / 2

        x = max(0, int(x - margin_w + offset_w))
        y = max(0, int(y - margin_h + offset_h))
        w = min(frame.shape[1], int(w + margin_w))
        h = min(frame.shape[0], int(h + margin_h))

        if save_images:
            save_frame(frame, (x, y, w, h))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face found", frame)


def process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.25,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) > 0:
        return max(faces, key=lambda f: f[2])  # return the widest face

    return None


def capture(video_capture):
    while True:
        ret, frame = video_capture.read()
        yield frame


def main(save_images=False):
    video_capture = cv2.VideoCapture(0)

    try:
        for frame in capture(video_capture):
            face = process(frame)
            show(frame, face, rect_scale=1.6, save_images=save_images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info('Stopping the video capturing.')

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture faces of people")
    parser.add_argument('-s', '--save', dest='save_images', action='store_true', default=False)
    args = parser.parse_args()

    main(save_images=args.save_images)
