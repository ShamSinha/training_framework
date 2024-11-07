from cxr_training.data.transforms.transform_controller import (
    RandAugment,
    Resize_Scale,
    Simple,
    Default,
)


class Base_transform:
    def __init__(self, args, mode) -> None:
        self.args = args
        self.mode = mode
        self.im_size = args.params.im_size
        self.train_transform = None
        self.val_transform = None
        self.set_transform()

    def set_transform(self):
        self.train_transform = RandAugment(self.args)
        self.val_transform = Resize_Scale(self.args)

    def apply_transformations(self, original_image, mask_dict):
        transform_images = None
        if self.mode == "train":
            transform_images, mask_dict = self.train_transform(
                original_image, mask_dict
            )
        elif self.mode == "val":
            transform_images, mask_dict = self.val_transform(original_image, mask_dict)

        return transform_images, mask_dict


class Default_transform:
    def __init__(self, args, mode) -> None:
        self.args = args
        self.mode = mode
        self.im_size = args.params.im_size
        self.train_transform = None
        self.val_transform = None
        self.set_transform()

    def set_transform(self):
        self.train_transform = Default(self.args)
        self.val_transform = Resize_Scale(self.args)

    def apply_transformations(self, original_image, mask_dict=None):
        transform_images = None
        if self.mode == "train":
            transform_images, mask_dict = self.train_transform(
                original_image, mask_dict
            )
        elif self.mode == "val":
            transform_images, mask_dict = self.val_transform(original_image, mask_dict)

        return transform_images, mask_dict
