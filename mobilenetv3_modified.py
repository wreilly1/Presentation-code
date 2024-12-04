import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define helper functions and model components
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
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
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
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
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
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


def mobilenetv3_small(**kwargs):
    cfgs = [
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


# Data loader function
def load_validation_data(root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(root=root, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader


# Test-Time Augmentation
# Efficient Test-Time Augmentation
def efficient_tta(model, inputs, device):
    """
    Apply efficient Test-Time Augmentation (TTA) with minimal runtime impact.

    Args:
        model: The PyTorch model to evaluate.
        inputs: Batch of input images.
        device: Device to run the model (CPU or CUDA).

    Returns:
        Averaged predictions from augmentations.
    """
    model.eval()
    with torch.no_grad():
        # Define lightweight augmentations
        augmentations = [
            inputs,  # Original image
            torch.flip(inputs, dims=[-1]),  # Horizontal flip
        ]
        # Run the model on each augmented version
        predictions = [model(aug.to(device)) for aug in augmentations]
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)


# Evaluation function with Efficient TTA
def evaluate_with_efficient_tta(model, loader, device, confidence_threshold=0.7, k=5):
    """
    Evaluates the model with Efficient Test-Time Augmentation (TTA).

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the validation dataset.
        device: Device to perform evaluation on (CPU or CUDA).
        confidence_threshold: Minimum confidence level for accepting predictions.
        k: Number of top predictions to consider for top-k accuracy.

    Returns:
        batch_indices: List of batch indices.
        top1_accuracies: List of top-1 accuracies over time.
        topk_accuracies: List of top-k accuracies over time.
        overall_top1_accuracy: Final top-1 accuracy as a percentage.
        overall_topk_accuracy: Final top-k accuracy as a percentage.
        confident_accuracy: Accuracy for confident predictions.
    """
    model.eval()
    total_samples = 0
    correct_top1 = 0
    correct_topk = 0
    confident_samples = 0
    confident_correct_top1 = 0

    batch_indices = []
    top1_accuracies = []
    topk_accuracies = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Evaluating with Efficient TTA", unit="batch")):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply Efficient TTA
            outputs = efficient_tta(model, inputs, device)

            # Calculate probabilities and apply confidence threshold
            probs = F.softmax(outputs, dim=1)
            max_probs, top1_pred = probs.max(dim=1)
            confident_mask = max_probs > confidence_threshold

            # Count confident samples
            confident_samples += confident_mask.sum().item()

            # Correct predictions for confident samples
            confident_correct_top1 += (top1_pred[confident_mask] == labels[confident_mask]).sum().item()

            # Top-1 Accuracy
            correct_top1 += (top1_pred == labels).sum().item()

            # Top-K Accuracy
            _, topk_pred = outputs.topk(k, dim=1)
            correct_topk += (topk_pred == labels.view(-1, 1)).sum().item()

            total_samples += labels.size(0)

            # Track batch-wise accuracies
            batch_top1_accuracy = (correct_top1 / total_samples) * 100
            batch_topk_accuracy = (correct_topk / total_samples) * 100
            top1_accuracies.append(batch_top1_accuracy)
            topk_accuracies.append(batch_topk_accuracy)
            batch_indices.append(batch_idx + 1)

    overall_top1_accuracy = (correct_top1 / total_samples) * 100
    overall_topk_accuracy = (correct_topk / total_samples) * 100
    confident_accuracy = (confident_correct_top1 / confident_samples * 100) if confident_samples > 0 else 0.0

    return batch_indices, top1_accuracies, topk_accuracies, overall_top1_accuracy, overall_topk_accuracy, confident_accuracy


# Accuracy Plotting
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


# Main execution
if __name__ == "__main__":
    val_root = r"C:\\Users\\Bigbi\\PycharmProjects\\pythonProject3\\mobilenetv3.pytorch-master\\extract_ImageNet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_small = mobilenetv3_small()
    net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))

    net_small = net_small.to(device)
    val_loader = load_validation_data(val_root, batch_size=32)

    print("\nEvaluating MobileNetV3-Small with Efficient TTA:")
    batch_indices, top1_acc, topk_acc, overall_top1_acc, overall_topk_acc, confident_acc = evaluate_with_efficient_tta(
        net_small, val_loader, device, confidence_threshold=0.7, k=5
    )

    # Print overall accuracies
    print(f"\nOverall Top-1 Accuracy: {overall_top1_acc:.2f}%")
    print(f"Overall Top-5 Accuracy: {overall_topk_acc:.2f}%")
    print(f"Confident Accuracy (Threshold > 0.7): {confident_acc:.2f}%")

    # Plot batch-wise accuracies
    print("\nPlotting Accuracy Over Time:")
    plot_accuracy_over_time(batch_indices, top1_acc, topk_acc)
