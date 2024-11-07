import cxr_training.nnmodule.models.Multihead as multihead_module
import cxr_training.nnmodule.models.CBAM_head as cbam_module


def get_model(args):
    """
    Some models like Swin and ConvNext perform a downsampling of scale=4 in the
    first block (stem) and then downsample by 2 afterwards with only depth=4 blocks.
    This results in an output size of half after the decoder.
    To get the same output size as the input you can pass head_upsampling=2
    which will upsample once more prior to the segmentation head.
    """
    enc = args.model.encoder
    dec = args.model.decoder
    in_ch = args.model.in_channels
    out_ch = args.model.out_channels
    recipe = args.trainer.recipe
    encoder_library = args.model.get('encoder_library', 'timm')


    if 'timm' in encoder_library:
        import torchseg
        DECODER_CLASSES = {
            "deeplabv3": torchseg.DeepLabV3,
            "deeplabv3plus": torchseg.DeepLabV3Plus,
            "fpn": torchseg.FPN,
            "linknet": torchseg.Linknet,
            "manet": torchseg.MAnet,
            "pan": torchseg.PAN,
            "pspnet": torchseg.PSPNet,
            "unet": torchseg.Unet,
            "unetplusplus": torchseg.UnetPlusPlus,
        }
        metadata = torchseg.encoders.TIMM_ENCODERS[enc]
        print(f"{enc}\n{metadata}")

        common_args = {
            "encoder_name": enc,
            "in_channels": in_ch,
            "classes": out_ch,
            "encoder_weights": True,
            "encoder_depth": len(metadata["channels"]),
            "encoder_params": {
                    "img_size": args.params.im_size,
                },
        }

        if dec.lower() in ["unet", "unetplusplus", "manet"]:
            if common_args["encoder_depth"] == 4:
                dec_chan = (256, 128, 64, 32)
            elif common_args["encoder_depth"] == 5:
                dec_chan = (256, 128, 64, 32, 16)
            common_args["decoder_channels"] = dec_chan

        elif dec.lower() in ["deeplabv3", "deeplabv3plus"]:
            dec_chan = 256
            common_args["decoder_channels"] = dec_chan

        if "swinv2" in enc.lower():
            # need to define img size since swin is a ViT hybrid
            common_args["encoder_params"] = {"img_size": args.params.im_size}
            smp_arch = DECODER_CLASSES[dec.lower()](**common_args)
        else:
            smp_arch = DECODER_CLASSES[dec.lower()](**common_args)

    elif 'smp' in encoder_library:
        import segmentation_models_pytorch as smp
        smp_arch = smp.create_model(arch=dec, encoder_name=enc, in_channels=in_ch, classes=out_ch)
        #smp_arch = smp.UnetPlusPlus(encoder_name=enc, in_channels=in_ch, classes=out_ch, encoder_weights="imagenet")
    
    model = multihead_module.MultiHead(smp_arch, args)

    # this is done to prevent an issue of gradients being none and
    # an issue with ddp which dosent allow gradients to go none

    if "seg" not in recipe:
        delattr(model.main_arch, "decoder")
    delattr(model.main_arch, "segmentation_head")

    return model


def get_main_model_string():
    multihead_file_path = multihead_module.__file__
    CBAM_file_path = cbam_module.__file__

    # Read the content of the MultiHead file
    model_str = open(multihead_file_path, "r").read()

    # Read the content of the CBAM file and append to model_str
    model_str += "\n\n" + open(CBAM_file_path, "r").read()

    return model_str