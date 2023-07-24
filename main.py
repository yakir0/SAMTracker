import time

import cv2
import numpy as np

from SAM import FSAM, FacebookSAM, MobileSAM
from utils import Bbox, draw_rect, draw_mask

save = False
display = True
debug = True
# save = display = False
benchmark = not save and not display and not debug

MAX_AGE = 20
IOU_THRESH = 0.4

# video_file = 'dog'
video_file = 'surfer'
# video_file = 'traffic'
# video_file = 'traffic2'

device = 'cpu'
# sam = FSAM(small=True, device=device)
# sam = FacebookSAM(model="vit_l", device=device)
sam = MobileSAM()

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

bboxes = cv2.selectROIs("Object Selection", original_image, fromCenter=False, showCrosshair=True)
bboxes = [Bbox(*bbox) for bbox in bboxes]
# bbox = Bbox(640, 380, 815, 550)  # dog
# bbox = Bbox(465, 282, 229, 200)  # surfer

sam.set_image(image_rgb)
prev_bboxes = []
for bbox in bboxes:
    ann = sam.box(bbox)
    prev_bbox = Bbox.from_mask(ann)
    prev_bboxes.append([prev_bbox, 0])

    if save or display:
        if debug:
            original_image = draw_mask(original_image, ann, (0, 0, 255))
            original_image = draw_rect(original_image, prev_bbox, (255, 0, 0))
        original_image = draw_rect(original_image, bbox, (0, 255, 0))

if save:
    result.write(original_image)

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
    sit = time.time()
    sam.set_image(image_rgb)
    print(f'took: {time.time() - sit} sec')

    for i, (prev_bbox, dead_frames) in enumerate(prev_bboxes):
        if dead_frames <= MAX_AGE:
            ann = sam.box(prev_bbox)
            ann_bbox = Bbox.from_mask(ann)

            if ann_bbox and prev_bbox.iou(ann_bbox) > IOU_THRESH:
                prev_bboxes[i][0] = ann_bbox
                prev_bboxes[i][1] = 0
                if display or save:
                    if debug:
                        original_image = draw_mask(original_image, ann, (0, 0, 255))
                        original_image = draw_rect(original_image, prev_bbox, (255, 0, 0))
                    original_image = draw_rect(original_image, ann_bbox, (0, 255, 0))
            else:
                prev_bboxes[i][1] += 1

    frame_count += 1
    if save:
        result.write(original_image)
    if display:
        cv2.imshow('Object Detection', original_image)
        # Press Q on keyboard to exit
        # key = cv2.waitKey(0)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
t2 = time.time()
fps = frame_count / (t2 - t1)
print(f'FPS: {fps}')
print(f'SFP: {1/fps}')

# When everything done, release the video capture object
video_capture.release()
if save:
    result.release()
# Closes all the frames
cv2.destroyAllWindows()
