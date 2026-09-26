"""
Data augmentation transforms for DINOv2 training on pathology images.
"""

import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class Random90Rotation(object):
    """
    With probability p, rotate the image by a discrete angle drawn
    uniformly from {0, 90, 180, 270}. Used under the pathology-FM recipe
    where tiles have no canonical orientation.

    Source: Virchow2, RudolfV, Hibou, Lunit all adopt 90-degree rotations.
    """
    ANGLES = (0, 90, 180, 270)

    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice(self.ANGLES)
            if angle != 0:
                img = TF.rotate(img, angle)
        return img


class TMEDinoTransforms(object):
    """
    DINO-style augmentation transforms for pathology images.

    When use_pathology_recipe=False (default), behaves exactly as before:
    every tile is pre-resized to (global_size, global_size) and then
    RandomResizedCrop is applied.

    When use_pathology_recipe=True, the pre-resize is dropped and the
    transform branches per-tile:
      - 40x source tiles (img.size[0] >= 448): ECT branch with probability
        ect_probability, standard crop-and-resize otherwise.
      - 20x source tiles (img.size[0] < 448): standard crop-and-resize
        only.
    The recipe also adds V-flip and 90-degree rotations to the color-jitter
    chain, and removes solarization from global_2 (Virchow2 Section 5.2
    ablation).

    Args:
        local_size: Size of local crops
        global_size: Size of global crops
        local_crop_scale: Scale range for local crops
        global_crop_scale: Scale range for global crops
        n_local_crops: Number of local crops
        mean: Normalization mean
        std: Normalization std
        use_pathology_recipe: Enable pathology-FM recipe augmentation path
        ect_probability: P(ECT branch | 40x tile) when recipe is enabled
    """
    def __init__(
        self,
        local_size=96,
        global_size=224,
        local_crop_scale=(0.05, 0.32),
        global_crop_scale=(0.32, 1.0),
        n_local_crops=2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        use_pathology_recipe=False,
        ect_probability=0.4,
        emit_scout=False,
        scout_pool_mode=False,
    ):
        self.n_local_crops = n_local_crops
        self.global_size = global_size
        self.local_size = local_size
        self.local_crop_scale = local_crop_scale
        self.global_crop_scale = global_crop_scale
        self.mean = mean
        self.std = std
        self.use_pathology_recipe = use_pathology_recipe
        self.ect_probability = ect_probability
        self.emit_scout = emit_scout
        # scout_pool_mode (stream thinning, GPU-augment path): emit ONLY the raw Resize(global_size)
        # uint8 tile -- NO augmentation. The trainer normalizes it for the scout/bank and GPU-augments
        # (kornia) only the thinned survivors, so the 5/6 of the pool that gets discarded is never
        # augmented. This supersedes emit_scout for thinned mode. weighted/off leave it False and get
        # the exact CPU crop pipeline below (byte-identical).
        self.scout_pool_mode = scout_pool_mode
        self.raw_resize = transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC)
        self.pil_to_uint8 = transforms.PILToTensor()   # uint8 [3,H,W] in 0-255, no /255, no normalize

        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose([
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        # Scout crop (stream thinning, section 3.9): an UNAUGMENTED normalized global view --
        # Resize(global_size) + ToTensor + Normalize, no RandomResizedCrop / flip / rotate /
        # ColorJitter / grayscale / blur. It is appended LAST in __call__ only when emit_scout is
        # set (thinned mode), and consumes NO RNG, so g1/g2/locals draw the identical random
        # augmentation they would without it -- weighted/off runs (emit_scout=False) are byte-
        # identical. The scout feeds the density bank only; it never enters the DINO/iBOT forwards.
        self.scout = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        if use_pathology_recipe:
            # V-flip and 90-degree rotations added; solarization removed from
            # global_2 below. Sources: Virchow2, RudolfV, Hibou, Lunit all
            # adopt V-flip and 90-degree rotations; solarization off from
            # Virchow2 Section 5.2 ablation.
            self.flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                Random90Rotation(p=0.75),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ])

            # ECT primitives: native 40x +/- 10%, used on 40x source tiles
            self.ect_global_1 = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=global_size, scale=(0.203, 0.303), ratio=(0.95, 1.05),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.GaussianBlur(9, (0.1, 2.0)),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            self.ect_global_2 = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=global_size, scale=(0.203, 0.303), ratio=(0.95, 1.05),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.1),
                # No RandomSolarize - Virchow2 Section 5.2 ablation
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            self.ect_local = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=local_size, scale=(0.037, 0.056), ratio=(0.95, 1.05),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.5),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            # Standard primitives: canonical DINOv2 ranges per Virchow2 5.1
            self.std_global_1 = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=global_size, scale=(0.32, 1.0), ratio=(0.75, 1.33),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.GaussianBlur(9, (0.1, 2.0)),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            self.std_global_2 = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=global_size, scale=(0.32, 1.0), ratio=(0.75, 1.33),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.1),
                # No RandomSolarize - Virchow2 Section 5.2 ablation
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            self.std_local = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=local_size, scale=(0.05, 0.32), ratio=(0.75, 1.33),
                    interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.5),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            # Original DINO color augmentation
            self.flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])

            # Global view 1
            self.global_1 = transforms.Compose([
                transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.GaussianBlur(9, (0.1, 2.0)),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            # Global view 2
            self.global_2 = transforms.Compose([
                transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

            # Local crops
            self.local = transforms.Compose([
                transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(size=local_size, scale=local_crop_scale, interpolation=Image.BICUBIC),
                self.flip_and_color_jitter,
                transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.5),
                self.to_tensor,
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, x):
        """
        Apply augmentation transforms.

        When use_pathology_recipe is True:
        - 40x tiles (img size >= 448) get ECT with probability ect_probability,
          standard crop-and-resize otherwise.
        - 20x tiles (img size < 448) always get standard crop-and-resize.

        When use_pathology_recipe is False, falls back to the original
        pre-resize-to-global_size + RandomResizedCrop behavior for all tiles.

        Returns:
            List of augmented crops: [global1, global2, local1, ..., localN]
            OR, in scout_pool_mode, [raw_uint8]: a single Resize(global_size) uint8 tile (no aug).
        """
        if self.scout_pool_mode:
            # Thinned GPU-augment path: return ONLY the raw resized uint8 tile. The trainer derives
            # the scout (normalize) for the bank and GPU-augments the survivors. Cheap: no crop
            # generation, no photometric ops -- the 5/6 discarded by thinning cost nothing here.
            return [self.pil_to_uint8(self.raw_resize(x))]

        crops = []

        if self.use_pathology_recipe:
            source_size = x.size[0]  # PIL image size[0] is width

            if source_size >= 448:
                # 40x tile - probabilistic ECT routing
                if random.random() < self.ect_probability:
                    g1, g2, lc = self.ect_global_1, self.ect_global_2, self.ect_local
                else:
                    g1, g2, lc = self.std_global_1, self.std_global_2, self.std_local
            else:
                # 20x tile - standard only (no extended context available)
                g1, g2, lc = self.std_global_1, self.std_global_2, self.std_local

            crops.append(g1(x))
            crops.append(g2(x))
            for _ in range(self.n_local_crops):
                crops.append(lc(x))
        else:
            # Original behavior, preserved for when recipe is off
            crops.append(self.global_1(x))
            crops.append(self.global_2(x))
            for _ in range(self.n_local_crops):
                crops.append(self.local(x))

        # Thinned mode only: append the unaugmented scout crop LAST (trailing index), after all
        # augmented crops have drawn their RNG. Deterministic (Resize+Normalize), so it does not
        # perturb the augmented crops' random state.
        if self.emit_scout:
            crops.append(self.scout(x))

        return crops
