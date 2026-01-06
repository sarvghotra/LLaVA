import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_and_transforms
from open_clip.transform import PreprocessCfg


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.args = args

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        class ImageProcessorWrapper:
            def __init__(self, img_processor, image_size):
                self.img_processor = img_processor

                if isinstance(image_size, tuple):
                    height = image_size[0]
                    width = image_size[1]
                else:
                    height, width = image_size, image_size
                self.crop_size = {
                    "height": height,
                    "width": width
                }

            def __call__(self):
                return self.img_processor

            def preprocess(self, image, return_tensors='.pt'):
                enc = self.img_processor(image)
                ret_load = {
                    'pixel_values': [enc]
                }
                return ret_load


        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower, image_processor = self.create_openclip_model_transform(self.vision_tower_name, self.args)
        pp_cfg = PreprocessCfg(**self.vision_tower.visual.preprocess_cfg)
        image_size = pp_cfg.size
        self.image_processor = ImageProcessorWrapper(image_processor, image_size)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        # Add a shim config so your existing properties work
        if not hasattr(self.vision_tower, 'config'):
            class OpenCLIPConfig:
                def __init__(self, model):
                    self.hidden_size = model.visual.width   # FIXME (critical): adhoc way to match the shapes. Ideally, it should not be here.
                    self.image_size = model.visual.image_size[0]
                    self.patch_size = model.visual.patch_size[0]

            self.vision_tower.config = OpenCLIPConfig(self.vision_tower)

    def create_openclip_model_transform(self, model_path, args):
        model_kwargs = {}
        # if args.siglip:
        #     model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        #     model_kwargs['init_logit_bias'] = -10
        if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
            # arg is nargs, single (square) image size list -> int
            args.force_image_size = args.force_image_size[0]

        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.vision_model_name,
            model_path,
            precision=args.precision,
            device=args.training_args_device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            force_context_length=args.force_context_length,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,  # only effective for inference
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            cache_dir=args.vision_cache_dir,
            strict_weight_load=(not args.non_strict_weight_load),
            **model_kwargs,
        )

        # model, preprocess_train, preprocess_val = create_model_and_transforms(
        #         "SEM_ViT-B-16",
        #         "/home/mila/s/sarvjeet-singh.ghotra/scratch/git/open_clip/src/logs/SEM_ViT-B-32_laion4/checkpoints/epoch_7.pt",
        #         precision="amp_bf16",
        #         device=0,
        #         jit=False,
        #         force_quick_gelu=False,
        #         force_custom_text=False,
        #         force_patch_dropout=None,
        #         force_image_size=None,
        #         force_context_length=None,
        #         image_mean=None,
        #         image_std=None,
        #         image_interpolation=None,
        #         image_resize_mode=None,
        #         aug_cfg={},
        #         pretrained_image=False,
        #         output_dict=True,
        #         cache_dir=None,
        #         strict_weight_load=True,
        #         **{},
        #     )

        return model, preprocess_train

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer + 1]
        # FIXME: commented it to match the shapes with projector layer.
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))  # , output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower.visual.forward_intermediates(images.to(device=self.device, dtype=self.dtype), output_fmt='NLC')
            # image_forward_outs = self.vision_tower.encode_image(images.to(device=self.device, dtype=self.dtype), normalize=False)  # , output_hidden_states=True)
            image_forward_outs = image_forward_outs['image_intermediates']
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # return self.vision_tower.dtype
        model_dtype = next(self.vision_tower.parameters()).dtype
        return model_dtype

    @property
    def device(self):
        # return self.vision_tower.device
        model_device = next(self.vision_tower.parameters()).device
        return model_device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
