import os.path
import time

import cv2
import numpy as np

from SAM import FSAM, FacebookSAM, MobileSAM
from utils import Bbox, draw_rect, draw_mask

MAX_AGE = 20
THRESH = 0.6
MIN_AREA = 20


def track(sam, video_file, *, bboxes=None, save=False, display=True, debug=True):
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

    if not bboxes:
        bboxes = cv2.selectROIs("Object Selection", original_image, fromCenter=False, showCrosshair=True)
        bboxes = [Bbox(*bbox) for bbox in bboxes]

    sam.set_image(image_rgb)
    prev_bboxes = []
    for bbox in bboxes:
        prev_bboxes.append([bbox, 0])

        if save or display:
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
        if any(dead_frames <= MAX_AGE for _, dead_frames in prev_bboxes):
            sit = time.time()
            sam.set_image(image_rgb)
            print(f'took: {time.time() - sit} sec')

            for i, (prev_bbox, dead_frames) in enumerate(prev_bboxes):
                if dead_frames <= MAX_AGE:
                    ann, score = sam.box(prev_bbox)
                    ann_bbox = Bbox.from_mask(ann)

                    df_ratio = (MAX_AGE - dead_frames)/MAX_AGE

                    if ann_bbox and ann_bbox.area > MIN_AREA and \
                            sum((prev_bbox.iou(ann_bbox), score, df_ratio))/3 > THRESH:
                        prev_bboxes[i][0] = ann_bbox
                        prev_bboxes[i][1] = 0
                        if display or save:
                            if debug:
                                original_image = draw_mask(original_image, ann, (255, 255, 0))
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
    print(f'SFP: {1 / fps}')

    # When everything done, release the video capture object
    video_capture.release()
    if save:
        result.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def track2(sam, video_capture, bboxes):
    ret, image = video_capture.read()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam.set_image(image_rgb)
    prev_bboxes = []
    for bbox in bboxes:
        ann = sam.box(bbox)
        prev_bbox = Bbox.from_mask(ann)
        prev_bboxes.append([prev_bbox, 0])

    t1 = time.time()
    frame_count = 0
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect
        if any(dead_frames <= MAX_AGE for _, dead_frames in prev_bboxes):
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
                    else:
                        prev_bboxes[i][1] += 1

        frame_count += 1
    t2 = time.time()
    fps = frame_count / (t2 - t1)
    print(f'FPS: {fps}')
    print(f'SFP: {1 / fps}')

    video_capture.release()


def output_results(sams, files, bboxes):
    for sam_cls, sam_args in sams:
        sam = sam_cls(**sam_args)
        for video_file in files:
            track(sam, video_file, bboxes=bboxes[video_file], save=True, debug=False)


def get_bboxes(files, save_new=False):
    res = {}
    for video_file in files:
        if os.path.exists(f'input/{video_file}.txt'):
            bboxes = np.genfromtxt(f'input/{video_file}.txt', delimiter=',', ndmin=2)
        else:
            video_capture = cv2.VideoCapture(f'videos/{video_file}.mp4')
            ret, image = video_capture.read()
            bboxes = cv2.selectROIs("Object Selection", image, fromCenter=False, showCrosshair=True)
            if save_new:
                np.savetxt(f'input/{video_file}.txt', bboxes, delimiter=', ', fmt='%i')
        res[video_file] = [Bbox(*bbox) for bbox in bboxes]
    return res



def main():
    save = False
    display = True
    debug = True

    video_files = [
        'dog',
        'surfer',
        'traffic',
        'traffic2',
        'lions',
        'peppers',
        'bikes',
    ]

    sams = [
        # (FSAM, {'small': True}),
        # (FSAM, {'small': False}),
        # (FacebookSAM, {'model': 'vit_b'}),
        # (FacebookSAM, {'model': 'vit_l'}),
        # (FacebookSAM, {'model': 'vit_h'}),
        (MobileSAM, {}),
    ]
    # sam = FacebookSAM(model="vit_l", device=device)
    # sam = MobileSAM()

    bbs = get_bboxes(video_files, True)
    output_results(sams, video_files, bbs)
    # track(sam, video_files[0], bboxes=bbs[video_files[0]], save=False, debug=True, display=True)


if __name__ == '__main__':
    main()
