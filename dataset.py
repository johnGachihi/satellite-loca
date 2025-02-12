from typing import Optional, Tuple, Callable, Union, Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image


def init_patch_matching_tracker(size: int = 14):
    """Initialize square grid to track patches correspondances in a mask."""
    return torch.reshape(torch.arange(size ** 2), (1, size, size))


# TODO: Inline this
def init_box_tracker():
    """Initialize box tracker."""
    return torch.zeros(5)


class CropFlipGenerateMask:
    """Crop and flip an image while tracking operations with a mask."""

    def a(self):
        print("")

    def __init__(
            self,
            resize_size: int = 224,
            area_min: int = 5,
            area_max: int = 100,
            flip: bool = True,
            resize_method: TF.InterpolationMode = TF.InterpolationMode.BILINEAR
    ):
        """
        Args:
            resize_size: Final size after crop and resize
            area_min: Minimum crop area as percentage of original image
            area_max: Maximum crop area as percentage of original image
            flip: Whether to apply random horizontal flip
        """
        self.resize_size = resize_size
        self.area_min = area_min
        self.area_max = area_max
        self.flip = flip
        self.resize_method = resize_method

    def __call__(self, image, mask):
        """

        Args:
            image: [C, H, W]
            mask:

        Returns:

        """
        _, original_h, original_w = image.shape

        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            image,
            scale=[self.area_min / 100, self.area_max / 100],
            ratio=[1.0, 1.0],
        )

        # Form bounding box
        bbox = torch.Tensor([0, i, j, h, w], device=image.device)

        # Crop and resize
        crop = TF.resized_crop(
            img=image, top=i, left=j, height=h, width=w,
            size=[self.resize_size, self.resize_size],
            interpolation=self.resize_method)

        if self.flip:
            do_flip = torch.zeros(1).uniform_().item() > 0.5
            if do_flip:
                crop = TF.hflip(crop)
                mask = TF.hflip(mask)

        mask = TF.resize(
            img=mask, size=[h, w], interpolation=TF.InterpolationMode.NEAREST)

        # Pad mask to match image size
        paddings = [j, original_w - w - j, i, original_h - h - i, 0, 0]
        mask = F.pad(mask, paddings, mode='constant', value=-1)

        return crop, mask, bbox


class ValueRange:
    """Transforms a [in_min,in_max] image to [vmin,vmax] range.

      Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
      channels independently.

      Args:
        vmin: A scalar. Output max value.
        vmax: A scalar. Output min value.
        in_min: A scalar or a list of input min values to scale. If a list, the
          length should match to the number of channels in the image.
        in_max: A scalar or a list of input max values to scale. If a list, the
          length should match to the number of channels in the image.
        clip_values: Whether to clip the output values to the provided ranges.
    """

    def __init__(self, vmin: float, vmax: float, in_min: float = 0, in_max: float = 255, clip_values: bool = True):
        self.vmin = vmin
        self.vmax = vmax
        self.in_min = in_min
        self.in_max = in_max
        self.clip_values = clip_values

    def __call__(self, image):
        in_min = self.in_min
        in_max = self.in_max

        # Scale to [0, 1]
        image = (image.float() - in_min) / (in_max - in_min)

        # Scale to [vmin, vmax]
        image = self.vmin + image * (self.vmax - self.vmin)

        return image


class InceptionCropWithMask:
    """Applies inception-style random crop and resize to image and mask."""

    def __init__(
            self,
            resize_size: Optional[Tuple[int, int]] = None,
            area_min: float = 5,
            area_max: float = 100,
            resize_mask: Optional[Tuple[int, int]] = None,
            resize_method: TF.InterpolationMode = TF.InterpolationMode.BILINEAR
    ):
        """
        Args:
            resize_size: Target size (height, width) after crop
            area_min: Minimum crop area as percentage of original image
            area_max: Maximum crop area as percentage of original image
            resize_mask: Optional different size for mask resize
        """
        self.resize_size = resize_size
        self.area_min = area_min / 100
        self.area_max = area_max / 100
        self.resize_mask = resize_mask
        self.resize_method = resize_method

    def compute_box_params(self, box: torch.Tensor, i: int, j: int, h: int, w: int):
        """Compute relative box parameters.

        Args:
            box: Input box tensor [validity, y, x, h, w]
            i: Top coordinate of crop (y)
            j: Left coordinate of crop (x)
            h: Height of crop
            w: Width of crop

        Returns:
            Tensor containing [validity, relative_begin_y, relative_begin_x,
                             relative_size_y, relative_size_x]
        """
        box = box[1:]

        relative_begin = torch.stack([
            (max(float(i) - box[0], 0.) / box[2]),  # y: (crop_y - box_y) / box_h
            (max(float(j) - box[1], 0.) / box[3])  # x: (crop_x - box_x) / box_w
        ])

        # Compute relative sizes
        relative_size = torch.stack([
            # y: (min(crop_y + crop_h, box_y + box_h) - box_y) / box_h - relative_begin_y - box_y/box_h
            max(torch.tensor(0.), min(float(i + h), box[0] + box[2]) / box[2]
                - relative_begin[0] - box[0] / box[2]),
            # x: (min(crop_x + crop_w, box_x + box_w) - box_x) / box_w - relative_begin_x - box_x/box_w
            max(torch.tensor(0.), min(float(j + w), box[1] + box[3]) / box[3]
                - relative_begin[1] - box[1] / box[3])
        ])

        # Check if box is big enough (both dimensions > 10% overlap)
        valid = torch.tensor([float((relative_size > 0.1).all())])

        # Return [validity, relative_begin_y, relative_begin_x, relative_size_y, relative_size_x]
        return torch.cat([valid, relative_begin, relative_size])

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, box: torch.Tensor):
        """Applies inception-style crop and resize to image and mask.

        Args:
            image: Input image tensor [C, H, W]
            mask: Input mask tensor [H, W]
            box: Input box tensor [validity, y, x, h, w]
        """
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            image,
            scale=[self.area_min, self.area_max],
            ratio=[3. / 4., 4. / 3.]
        )

        image_cropped = TF.crop(image, i, j, h, w)
        if self.resize_size:
            image_cropped = TF.resize(
                image_cropped,
                self.resize_size,
                interpolation=self.resize_method,
                antialias=True
            )

        box = self.compute_box_params(box, i, j, h, w)
        _box = torch.tensor([0, i, j, h, w], device=image.device)

        mask_cropped = TF.crop(mask, i, j, h, w)
        if self.resize_size:
            mask_cropped = TF.resize(
                mask_cropped,
                self.resize_size,
                interpolation=TF.InterpolationMode.NEAREST
            )
        if self.resize_mask:
            mask_cropped = TF.resize(
                mask_cropped,
                self.resize_mask,
                interpolation=TF.InterpolationMode.NEAREST
            )

        return image_cropped, mask_cropped, box, _box


class RandomFlipWithMask:
    """Flip image and mask horizontally."""

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        """Flip image and mask horizontally.

        Args:
            image: Input image tensor [C, H, W]
            mask: Input mask tensor [H, W]
        """
        do_flip = torch.zeros(1).uniform_().item() > 0.5
        if do_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


class RandomColorJitter:
    def __init__(
            self,
            p: float = 1.0,
            brightness: float = 0.4,
            contrast: float = 0.4,
            saturation: float = 0.2,
            hue: float = 0.1
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: torch.Tensor):
        if torch.rand((1,)).item() < self.p:
            return torchvision.transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )(image)
        else:
            return image


class RandomGaussianBlur:
    def __init__(
            self,
            p: float = 1.0,
            kernel_size: int = 224 // 10 - 1,
            sigma: Sequence[float] = (0.1, 2.0)
    ):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: torch.Tensor):
        if torch.rand((1,)).item() < self.p:
            return torchvision.transforms.GaussianBlur(kernel_size=self.kernel_size)(image)
        else:
            return image


def image_loader():
    """Load image from path."""

    def _image_loader(image_path):
        image = Image.open(image_path)
        image = TF.to_tensor(image)
        return image

    return _image_loader


class LOCATransform:
    """Transform pipeline for LOCA"""

    MEAN_RGB = [0.485, 0.456, 0.406]
    STDDEV_RGB = [0.229, 0.224, 0.225]

    def __init__(
            self,
            reference_size: int = 224,
            ref_mask_size: int = 14,
            n_queries: int = 10,
            query_size: int = 96,
            query_mask_size: int = 6,
    ):
        self.mask_size = ref_mask_size
        self.n_queries = n_queries

        self.ref_crop_and_mask = CropFlipGenerateMask(
            resize_size=reference_size,
            area_min=32,
            area_max=100,
            flip=False,
        )
        self.query0_crop_and_mask = InceptionCropWithMask(
            resize_size=(224, 224),
            area_min=32,
            area_max=100,
            resize_mask=(ref_mask_size, ref_mask_size),
        )
        self.query_crop_and_mask = InceptionCropWithMask(
            resize_size=(query_size, query_size),
            area_min=5,
            area_max=32,
            resize_mask=(query_mask_size, query_mask_size),
        )

    def __call__(self, img: torch.Tensor):
        # Create and augment reference
        mask = init_patch_matching_tracker(size=self.mask_size)
        ref, ref_mask, ref_box = self.ref_crop_and_mask(img, mask)
        ref = ValueRange(0, 1)(ref)
        ref = RandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(ref)
        ref = torchvision.transforms.RandomGrayscale(p=0.2)(ref)
        ref = RandomGaussianBlur(p=1.0)(ref)
        ref = torchvision.transforms.Normalize(
            mean=LOCATransform.MEAN_RGB,
            std=LOCATransform.STDDEV_RGB
        )(ref)

        # Create and augment queries
        # Query 0 is distinct from other queries
        queries: list = [self.query0_crop_and_mask(img, ref_mask, ref_box)]

        # Create other queries: [1, n_queries)
        for i in range(1, self.n_queries):
            queries.append(self.query_crop_and_mask(img, ref_mask, ref_box))

        # Augmentation for all queries
        for i in range(self.n_queries):
            query, q_mask, q_rel_box, q_box = queries[i]
            query, q_mask = RandomFlipWithMask()(query, q_mask)
            query = ValueRange(0, 1)(query)
            query = RandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(query)
            query = torchvision.transforms.RandomGrayscale(p=0.2)(query)
            queries[i] = (query, q_mask, q_rel_box, q_box)

        # Augmentation for queries [1, n_queries)
        for i in range(1, self.n_queries):
            query, q_mask, q_rel_box, q_box = queries[i]
            query = RandomGaussianBlur(p=0.5)(query)
            queries[i] = (query, q_mask, q_rel_box, q_box)

        # Augmentation for query 0
        query0, q_mask0, q_rel_box, q_box0 = queries[0]
        query0 = RandomGaussianBlur(p=0.1)(query0)
        query0 = torchvision.transforms.RandomSolarize(0.2)(query0)
        queries[0] = (query0, q_mask0, q_rel_box, q_box0)

        # Augmentation for all queries again
        for i in range(self.n_queries):
            query, q_mask, q_rel_box, q_box = queries[i]
            query = torchvision.transforms.Normalize(
                mean=LOCATransform.MEAN_RGB,
                std=LOCATransform.STDDEV_RGB
            )(query)
            queries[i] = (query, q_mask, q_rel_box, q_box)

        return ref, ref_mask, ref_box, queries
