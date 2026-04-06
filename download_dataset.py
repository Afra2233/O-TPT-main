import os
from torchvision.datasets import (
    CIFAR10, CIFAR100, Food101, Flowers102, DTD,
    OxfordIIITPet, UCF101, FGVCAircraft, StanfordCars
)

ROOT = "/scratch/hpc/07/zhang303/O-TPT-main/data"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_all():
    ensure_dir(ROOT)

    # CIFAR
    CIFAR10(root=os.path.join(ROOT, "cifar10"), train=True, download=True)
    CIFAR10(root=os.path.join(ROOT, "cifar10"), train=False, download=True)

    CIFAR100(root=os.path.join(ROOT, "cifar100"), train=True, download=True)
    CIFAR100(root=os.path.join(ROOT, "cifar100"), train=False, download=True)

    # Food101
    Food101(root=os.path.join(ROOT, "Food101"), split="train", download=True)
    Food101(root=os.path.join(ROOT, "Food101"), split="test", download=True)

    # Flowers102
    Flowers102(root=os.path.join(ROOT, "Flower102"), split="train", download=True)
    Flowers102(root=os.path.join(ROOT, "Flower102"), split="val", download=True)
    Flowers102(root=os.path.join(ROOT, "Flower102"), split="test", download=True)

    # DTD
    # DTD(root=os.path.join(ROOT, "DTD"), split="train", download=True)
    # DTD(root=os.path.join(ROOT, "DTD"), split="val", download=True)
    # DTD(root=os.path.join(ROOT, "DTD"), split="test", download=True)

    # Oxford-IIIT Pet
    OxfordIIITPet(root=os.path.join(ROOT, "OxfordPets"), split="trainval", download=True)
    OxfordIIITPet(root=os.path.join(ROOT, "OxfordPets"), split="test", download=True)

    # FGVC Aircraft
    FGVCAircraft(root=os.path.join(ROOT, "fgvc_aircraft"), split="train", download=True)
    FGVCAircraft(root=os.path.join(ROOT, "fgvc_aircraft"), split="val", download=True)
    FGVCAircraft(root=os.path.join(ROOT, "fgvc_aircraft"), split="test", download=True)

    # # Stanford Cars
    # StanfordCars(root=os.path.join(ROOT, "StanfordCars"), split="train", download=True)
    # StanfordCars(root=os.path.join(ROOT, "StanfordCars"), split="test", download=True)

    # # UCF101
    # # 需要 video 数据和 annotation 目录；第一次下载会比较大
    # ucf_root = os.path.join(ROOT, "UCF101")
    # ann_root = os.path.join(ROOT, "UCF101_annotations")
    # ensure_dir(ucf_root)
    # ensure_dir(ann_root)

    # UCF101(
    #     root=ucf_root,
    #     annotation_path=ann_root,
    #     frames_per_clip=1,
    #     step_between_clips=1,
    #     train=True,
    #     fold=1,
    #     download=True,
    # )
    # UCF101(
    #     root=ucf_root,
    #     annotation_path=ann_root,
    #     frames_per_clip=1,
    #     step_between_clips=1,
    #     train=False,
    #     fold=1,
    #     download=True,
    # )

if __name__ == "__main__":
    download_all()
    print("All downloads finished.")