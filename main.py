#!/usr/bin/env python
import cv2
import os
import sys
import time


cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def get_cascades(directory='cascades'):
    cascades = []
    for root, dirs, files in os.walk(directory):
        cascades += filter(lambda f: f.endswith('.xml'), files)
    return cascades


def save(frame, face):
    if not face:
        return

    x, y, w, h = face
    cv2.imwrite('last_face.png', frame[x:x+w, y:y+h])


def show(frame, face, rect_scale=1.0):
    if face is not None:
        x, y, w, h = face

        margin_w = (w * rect_scale - w) / 2
        margin_h = (h * rect_scale - h)

        offset_w = 0
        offset_h = margin_h / 2

        x = max(0, int(x - margin_w + offset_w))
        y = max(0, int(y - margin_h + offset_h))
        w = min(frame.shape[1], int(w + margin_w))
        h = min(frame.shape[0], int(h + margin_h))

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


def main():
    video_capture = cv2.VideoCapture(0)

    try:
        for frame in capture(video_capture):
            face = process(frame)
            show(frame, face, rect_scale=1.6)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
