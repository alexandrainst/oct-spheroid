from src.models.model_templates import resnet_3D


def generate_model(cnn_name, model_depth, n_classes, in_channels, sample_size):
    # simple CNN
    if cnn_name == "resnet":
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = resnet_3D.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0,
        )
    else:
        raise NotImplementedError

    return model
