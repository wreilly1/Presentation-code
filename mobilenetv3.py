"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)

   def load_validation_data(root, batch_size=1024):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(root=root, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader

def evaluate_model_with_topk_over_time(model, loader, device, k=5):
    """
    Evaluates the model over time and tracks top-1 and top-k accuracy for each batch.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the validation dataset.
        device: Device to perform evaluation on (CPU or CUDA).
        k: The number of top predictions to consider for top-k accuracy.

    Returns:
        batch_indices: List of batch indices.
        top1_accuracies: List of top-1 accuracies over time.
        topk_accuracies: List of top-k accuracies over time.
        overall_top1_accuracy: Final top-1 accuracy as a percentage.
        overall_topk_accuracy: Final top-k accuracy as a percentage.
    """
    model.eval()
    top1_accuracies = []
    topk_accuracies = []
    batch_indices = []
    total_samples = 0
    correct_top1 = 0
    correct_topk = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Evaluating", unit="batch")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Top-1 predictions
            _, top1_pred = outputs.max(1)
            correct_top1 += (top1_pred == labels).sum().item()

            # Top-K predictions
            _, topk_pred = outputs.topk(k, dim=1)
            correct_topk += (topk_pred == labels.view(-1, 1)).sum().item()

            total_samples += labels.size(0)

            # Calculate accuracies
            top1_accuracy = correct_top1 / total_samples
            topk_accuracy = correct_topk / total_samples

            top1_accuracies.append(top1_accuracy * 100)
            topk_accuracies.append(topk_accuracy * 100)
            batch_indices.append(batch_idx + 1)

    overall_top1_accuracy = (correct_top1 / total_samples) * 100
    overall_topk_accuracy = (correct_topk / total_samples) * 100

    return batch_indices, top1_accuracies, topk_accuracies, overall_top1_accuracy, overall_topk_accuracy

def plot_accuracy_over_time(batch_indices, top1_accuracies, topk_accuracies):
    """
    Plots the accuracy of top-1 and top-k predictions over time.

    Args:
        batch_indices: List of batch indices.
        top1_accuracies: List of top-1 accuracies over time.
        topk_accuracies: List of top-k accuracies over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, top1_accuracies, label="Top-1 Accuracy", marker="o")
    plt.plot(batch_indices, topk_accuracies, label="Top-5 Accuracy", marker="s")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 and Top-5 Accuracy Over Time")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    val_root = r"C:\\Users\\Bigbi\\PycharmProjects\\pythonProject3\\mobilenetv3.pytorch-master\\extract_ImageNet"  # Directory with validation images organized into subfolders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_small = mobilenetv3_small()
    net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth', weights_only=True))

    net_small = net_small.to(device)
    val_loader = load_validation_data(val_root, batch_size=1024)

    # Get class names from the dataset
    class_names = val_loader.dataset.classes

    # Evaluate model and track accuracy over time
    print("\nEvaluating Pretrained MobileNetV3-Small:")
    batch_indices, top1_accuracies, topk_accuracies, overall_top1_accuracy, overall_topk_accuracy = evaluate_model_with_topk_over_time(net_small, val_loader, device, k=5)

    # Print overall accuracies
    print(f"\nOverall Top-1 Accuracy: {overall_top1_accuracy:.2f}%")
    print(f"Overall Top-5 Accuracy: {overall_topk_accuracy:.2f}%")

    # Plot top-1 and top-5 accuracy over time
    print("\nPlotting Accuracy Over Time:")
    plot_accuracy_over_time(batch_indices, top1_accuracies, topk_accuracies)



