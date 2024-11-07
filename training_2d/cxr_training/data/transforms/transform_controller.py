from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from cxr_training.data.transforms import transform_utils as new_tsfms
from torchvision.transforms import Compose
import albumentations as A
from numpy import int64
import numpy as np
from cv2 import INTER_AREA


def modify_mask(original_mask):
    flag = 0
    if original_mask.min() == -100:
        original_mask[original_mask == -100] = 2.0
        original_mask = original_mask.astype(dtype=np.float32)
        flag = 1
    else:
        original_mask = original_mask.astype(dtype=np.float32)
    return flag, original_mask


def mask_replay_augment(seg_classes, replay, mask_dict):
    for tag in seg_classes:
        original_mask = mask_dict["segmentation_target"][tag]

        """
        THis is done so that if we encounter invalid seg maps (-100,-100) then we
        convert them to 2 so that after the augmentation they are reverted back,
        this is important as if they are not done then the value will shift from
        -100 to something like -100.0001 and then this would cause an error or
        worse the loss would be propogated , it is important to make it
        dtype=int64 as after the augmentation 2.0 sometimes becomes 2.00005
        and it wont be converted back to orginal -100 value.
        """
        flag, original_mask = modify_mask(original_mask)

        tf_mask = A.ReplayCompose.replay(replay, image=original_mask)
        tf_mask_image = tf_mask["image"].astype(dtype=int64)

        if flag:
            tf_mask_image[tf_mask_image == 2.0] = -100
        else:
            tf_mask_image = (tf_mask_image > 0) * 1

        mask_dict["segmentation_target"][tag] = tf_mask_image.astype(dtype=int64)

    return mask_dict


class Test_Augment:
    def __init__(self, args) -> None:
        self.args = args
        self.im_size = args.params.im_size
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.Resize(
                    self.im_size,
                    self.im_size,
                    interpolation=INTER_AREA,
                    always_apply=True,
                )
            ]
        )

        self.transforms = Compose(
            [
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, idx=0):
        try:
            original_image = new_tsfms.scale(original_image)
            tf_image = self.albumentation_transforms(image=original_image)
            augmented_image = self.transforms(tf_image["image"])

        except Exception as e:
            print("\033[91m" + f"applying transform error in {idx}" + "\033[0m")
            print(e)
            return original_image

        return augmented_image


class TestTimeAugment:
    def __init__(self, args) -> None:
        self.args = args
        self.size = args.params.im_size
        x = np.random.uniform(-0.1, 0.1, size=1)
        y = np.random.uniform(-0.1, 0.1, size=1)
        self.t_percent = {"x": x[0], "y": y[0]}

        self.albumentation_transforms = A.ReplayCompose(
            [
                A.GridDistortion(distort_limit=0.3, p=1),
                A.transforms.GaussNoise(var_limit=(10, 50), p=1),
                A.Rotate(limit=15, p=1),
                A.Affine(translate_percent=self.t_percent, p=1),
                A.Resize(
                    self.size, self.size, interpolation=INTER_AREA, always_apply=True
                ),
            ]
        )

        self.transforms = Compose(
            [
                new_tsfms.scale,
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, idx=0):
        normal_transform = Test_Augment(self.args)
        test_image = normal_transform(original_image)

        augmentations = []
        augmentations.append(test_image)

        for _id in range(9):
            try:
                # original_image = new_tsfms.scale(original_image)
                tf_image = self.albumentation_transforms(image=original_image)

                augmented_image = self.transforms(tf_image["image"])
                augmentations.append(augmented_image)
            except Exception as e:
                print("\033[91m" + f"applying transform error in {idx}" + "\033[0m")
                print(e)
                augmentations.append(test_image)

        return augmentations


class Resize_Scale:
    def __init__(self, args) -> None:
        self.im_size = args.params.im_size
        self.recipe = args.trainer.recipe
        self.seg_heads = args.seg.heads
        self.cls_heads = args.cls.heads
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.Resize(
                    self.im_size,
                    self.im_size,
                    interpolation=INTER_AREA,
                    always_apply=True,
                )
            ]
        )

        self.transforms = Compose(
            [
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, mask_dict):
        try:
            original_image = new_tsfms.scale(original_image)
            tf_image = self.albumentation_transforms(image=original_image)
            augmented_image = self.transforms(tf_image["image"])
            if "seg" in self.recipe:
                mask_dict = mask_replay_augment(
                    self.seg_heads, tf_image["replay"], mask_dict
                )
        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image, mask_dict


class Default:
    def __init__(self, args) -> None:
        self.size = args.params.im_size
        self.recipe = args.trainer.recipe
        self.seg_heads = args.seg.heads
        self.cls_heads = args.cls.heads
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                            p=0.5,
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                A.Rotate(limit=90, p=0.2),
                A.Resize(1100, 1100, interpolation=INTER_AREA, always_apply=True),
                A.RandomCrop(960, 960, always_apply=True),
                A.Resize(
                    self.size, self.size, interpolation=INTER_AREA, always_apply=True
                ),
            ]
        )
        self.transforms = Compose(
            [
                new_tsfms.scale,
                new_tsfms.gamma,
                new_tsfms.brightness,
                new_tsfms.contrast,
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, mask_dict=None):
        try:
            original_image = new_tsfms.scale(original_image)
            tf_image = self.albumentation_transforms(image=original_image)
            augmented_image = self.transforms(tf_image["image"])

            if "seg" in self.recipe:
                mask_dict = mask_replay_augment(
                    self.seg_heads, tf_image["replay"], mask_dict
                )
        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image, mask_dict


class Simple:
    def __init__(self, args) -> None:
        self.size = args.params.im_size
        self.recipe = args.trainer.recipe
        self.seg_heads = args.seg.heads
        self.cls_heads = args.cls.heads
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.Resize(
                    self.size, self.size, interpolation=INTER_AREA, always_apply=True
                )
            ]
        )
        self.transforms = Compose(
            [
                new_tsfms.scale,
                new_tsfms.clip,
                new_tsfms.gamma,
                new_tsfms.brightness,
                new_tsfms.contrast,
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, mask_dict=None):
        try:
            original_image = new_tsfms.scale(original_image)
            tf_image = self.albumentation_transforms(image=original_image)
            augmented_image = self.transforms(tf_image["image"])

            if "seg" in self.recipe:
                mask_dict = mask_replay_augment(
                    self.seg_heads, tf_image["replay"], mask_dict
                )
        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image, mask_dict


class RandAugment:
    def __init__(self, args) -> None:
        self.size = args.params.im_size
        self.recipe = args.trainer.recipe
        self.seg_heads = args.seg.heads
        self.cls_heads = args.cls.heads
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                            p=0.5,
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.Sequential(
                            [
                                A.Resize(
                                    1100,
                                    1100,
                                    interpolation=INTER_AREA,
                                    always_apply=True,
                                ),
                                A.RandomCrop(960, 960, always_apply=True),
                            ]
                        ),
                        A.Rotate(limit=15, p=0.5),
                    ],
                    p=0.5,
                ),
                A.Resize(
                    self.size, self.size, interpolation=INTER_AREA, always_apply=True
                ),
            ]
        )

        self.transforms = Compose(
            [
                new_tsfms.scale,
                new_tsfms.clip,
                new_tsfms.gamma,
                new_tsfms.brightness,
                new_tsfms.contrast,
                new_tsfms.clip,
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image, mask_dict=None):
        try:
            original_image = new_tsfms.scale(original_image)
            tf_image = self.albumentation_transforms(image=original_image)

            augmented_image = self.transforms(tf_image["image"])

            if "seg" in self.recipe:
                mask_dict = mask_replay_augment(
                    self.seg_heads, tf_image["replay"], mask_dict
                )
        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image, mask_dict