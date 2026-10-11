import torch
import timm
from treemapmix import TreemapMix, PlackettLuceLoss


def main():
    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.arange(4)
    mixed, target = TreemapMix(k=4)(images, labels)
    model = timm.create_model("resnet18", pretrained=False, num_classes=4)
    loss = PlackettLuceLoss()(model(mixed), target)
    loss.backward()
    print(f"PL loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
