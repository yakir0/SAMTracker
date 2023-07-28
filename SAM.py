import cv2
import numpy as np
import torch
from utils import Bbox
from FastSAM import FastSAM, FastSAMPrompt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from mobile_sam import (
    SamPredictor as mb_SamPredictor,
    sam_model_registry as mb_sam_model_registry
)


class SAM:
    def __init__(self, name, device=None):
        self.name = name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def set_image(self, image):
        """

        Args:
            image: ndarray, RGB

        Returns: None

        """
        pass

    def box(self, bbox):
        """

        Args:
            bbox: Bbox

        Returns: ndarray of the size of the image with True/False values

        """
        pass


class FSAM(SAM):
    def __init__(self, small=True, device=None):
        model_name = f"FastSAM-{'s' if small else 'x'}"
        super().__init__(model_name, device)
        self.model = FastSAM(f'models/{model_name}.pt')
        self.prompt_process = None

    def set_image(self, image):
        everything_results = self.model(
            image,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.35,
            iou=0.80,
        )

        self.prompt_process = FastSAMPrompt(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            everything_results,
            device=self.device
        )

    def box(self, bbox: Bbox):
        return self.prompt_process.box_prompt(bbox=bbox.get_xyxy())[0].astype(bool)


class FacebookSAM(SAM):
    def __init__(self, model="default", device=None):
        models = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        models['default'] = models['vit_h']
        super().__init__(f'sam-{model}', device)
        sam = sam_model_registry[model](checkpoint=f"models/{models[model]}")
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def box(self, bbox):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox.get_xyxy())[None, :],
            multimask_output=True,
        )
        return masks[np.argmax(scores)]


class MobileSAM(SAM):
    def __init__(self, device=None):
        super().__init__(f'mobile-sam', device)
        sam = mb_sam_model_registry['vit_t'](checkpoint=f"models/mobile_sam.pt")
        sam.to(self.device)
        self.predictor = mb_SamPredictor(sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def box(self, bbox):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox.get_xyxy())[None, :],
            multimask_output=True,
        )
        return masks[np.argmax(scores)]
