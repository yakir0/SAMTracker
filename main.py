import time

import cv2
import numpy as np

from SAM import FSAM
from utils import Bbox

save = True
display = True
debug = True

video_file = 'dog'
# video_file = 'surfer'
# video_file = 'traffic2'

sam = FSAM()

# Initialize the video capture
video_capture = cv2.VideoCapture(f'videos/{video_file}.mp4')

if save:
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'output/{video_file}-{sam.name}-res.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             vid_fps, (frame_width, frame_height))

ret, image = video_capture.read()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_image = image.copy()

bbox = cv2.selectROI("Object Selection", original_image, fromCenter=False, showCrosshair=True)
bbox = Bbox(*bbox)

sam.set_image(image_rgb)
ann = sam.box(bbox)
prev_bbox = Bbox.from_mask(ann)

if save or display:
    if debug:
        masked_img = np.where(ann[..., None], [0, 0, 255], original_image)
        original_image = cv2.addWeighted(original_image, 0.8, masked_img, 0.2, 0, dtype=0)
        cv2.rectangle(original_image, (prev_bbox.x, prev_bbox.y), (prev_bbox.x2, prev_bbox.y2), (255, 0, 0), 2)
    cv2.rectangle(original_image, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 255, 0), 2)
if save:
    result.write(original_image)

# if display:
#     cv2.rectangle(original_image, (res_bbox.x, res_bbox.y), (res_bbox.x2, res_bbox.y2), (255, 0, 0), 2)
#     cv2.imshow("res", original_image)
#     cv2.waitKey(0)

t1 = time.time()
frame_count = 0
while True:
    ret, image = video_capture.read()
    if not ret:
        break
    if display or save:
        original_image = image.copy()  # Create a copy for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect
    sam.set_image(image_rgb)
    ann = sam.box(prev_bbox)
    ann_bbox = Bbox.from_mask(ann)

    frame_count += 1
    if display or save:
        if debug:
            masked_img = np.where(ann[..., None], [0, 0, 255], original_image)
            original_image = cv2.addWeighted(original_image, 0.8, masked_img, 0.2, 0, dtype=0)
            cv2.rectangle(original_image, (prev_bbox.x, prev_bbox.y), (prev_bbox.x2, prev_bbox.y2), (255, 0, 0), 2)
        cv2.rectangle(original_image, (ann_bbox.x, ann_bbox.y), (ann_bbox.x2, ann_bbox.y2), (0, 255, 0), 2)
    prev_bbox = ann_bbox
    if save:
        result.write(original_image)
    if display:
        cv2.imshow('Object Detection', original_image)
        # Press Q on keyboard to exit
        key = cv2.waitKey(0)
        # key = cv2.waitKey(0 if debug else 25)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            tmp = sam.everything()
            sam.prompt_process.plot(annotations=tmp,output='./output/tmp.jpg',)
            cv2.waitKey(0)
t2 = time.time()
print(f'FPS: {frame_count / (t2 - t1)}')

# When everything done, release the video capture object
video_capture.release()
if save:
    result.release()
# Closes all the frames
cv2.destroyAllWindows()
