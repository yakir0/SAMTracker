import cv2
import torch
from utils import Bbox
from FastSAM import FastSAM, FastSAMPrompt

class SAM:
    def __init__(self, name, device=None):
        self.name = name
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def set_image(self, image):
        pass

    def box(self, bbox):
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
            conf=0.4,
            iou=0.9,
        )

        self.prompt_process = FastSAMPrompt(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            everything_results,
            device=self.device
        )

    def box(self, bbox: Bbox):
        return self.prompt_process.box_prompt(bbox=bbox.get_xyxy())[0]

