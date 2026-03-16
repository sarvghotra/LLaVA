import torch
import torch.nn as nn

from .flair.src.flair.factory import create_model_and_transforms
from open_clip.transform import PreprocessCfg


class FlairVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        model, _, preprocess_val = create_model_and_transforms(
            self.args.vision_model_name,
            self.vision_tower_name,  # pretrained weights path
            precision=getattr(self.args, 'precision', 'bfp16'),
            device=self.args.training_args_device,
        )

        self.vision_tower = model
        self.vision_tower.requires_grad_(False)

        pp_cfg = PreprocessCfg(**self.vision_tower.visual.preprocess_cfg)
        image_size = pp_cfg.size
        self.image_processor = ImageProcessorWrapper(preprocess_val, image_size)

        # Attach a config-like object so existing properties work
        if not hasattr(self.vision_tower, 'config'):
            visual = self.vision_tower.visual
            self.vision_tower.config = _FlairVisionConfig(visual)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, list):
            image_forward_outs = image_forward_outs[self.select_layer + 1]
        return image_forward_outs

    def _forward_wo_text(self, image):
        global_image_token, local_image_tokens = self.vision_tower.encode_image(
                        image.to(device=self.device, dtype=self.dtype)
                    )
        local_image_tokens = self.feature_select(local_image_tokens).to(image.dtype)
        return local_image_tokens

    # def _forward_w_text(self, images):
    #     global_image_token, local_image_tokens = self.vision_tower.encode_image(
    #                     image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
    #                 )

    @torch.no_grad()
    def forward(self, images, texts=None):
        if type(images) is list:
            image_features = []
            if texts is None:
                for image in images:
                    image_features.append(self._forward_wo_text(image))
            else:
                for image, text in zip(images, texts):
                    image_features.append(self._forward_w_text(image, text))
        else:
            image_features = self._forward_w_text(images, texts) if texts is not None else self._forward_wo_text(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class _FlairVisionConfig:
    """Minimal config shim exposing the fields LLaVA expects."""
    def __init__(self, visual):
        self.hidden_size = visual.width
        image_size = visual.image_size
        self.image_size = image_size[0] if isinstance(image_size, (tuple, list)) else image_size
        patch_size = visual.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size


class ImageProcessorWrapper:
    """Wraps a torchvision-style transform to match the HuggingFace processor interface."""
    def __init__(self, transform, image_size):
        self.transform = transform
        if isinstance(image_size, (tuple, list)):
            h, w = image_size[0], image_size[1]
        else:
            h = w = image_size
        self.crop_size = {'height': h, 'width': w}

    def __call__(self, image, return_tensors='.pt'):
        if isinstance(image, list):
            assert len(image) == 1, 'Expected a single image'
            return self.preprocess(image[0], return_tensors)
        return self.preprocess(image, return_tensors)

    def preprocess(self, image, return_tensors='.pt'):
        return {'pixel_values': [self.transform(image)]}
