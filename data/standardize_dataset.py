import copy
import json


def dataset_prompt_template(dataset):
    if dataset in ["MNIST", "SVHN"]:
        templates = [
            'a photo of the number: "{}".',
        ]
    elif dataset in [
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "CUB200",
        "ImageNet",
        "ImageNetA",
        "ImageNetR",
        "TinyImageNet",
        "Caltech101",
        "Caltech256",
        "SUN397",
    ]:
        templates = [
            "a photo of a {}.",
            # 'a photo of the {}.',
        ]
    elif dataset == "Flowers102":
        templates = [
            "a photo of a {}, a type of flower.",
        ]
    elif dataset == "FGVCAircraft":
        templates = [
            "a photo of a {}, a type of aircraft.",
            # 'a photo of the {}, a type of aircraft.',
        ]
    elif dataset == "Food101":
        templates = [
            "a photo of {}, a type of food.",
        ]
    elif dataset == "StanfordCars":
        templates = [
            "a photo of a {}, a type of car.",
        ]
    elif dataset == "OxfordIIITPet":
        templates = [
            "a photo of a {}, a type of pet.",
        ]
    elif dataset == "DTD":
        templates = [
            "a photo of a {} texture.",
        ]
    elif dataset == "Places365LT":
        templates = [
            "a photo of the {}, a type of place.",
        ]
    elif dataset == "iNaturalist":
        templates = [
            "a photo of a {}, a type of species.",
        ]
    elif dataset == "PCAM":
        templates = [
            "this is a photo of {}",
        ]
    elif dataset == "EuroSAT":
        templates = [
            "a centered satellite photo of {}.",
        ]
    elif dataset == "Resisc45":
        templates = [
            "satellite photo of {}.",
        ]
    elif dataset == "UCF101":
        templates = [
            "a video of a person doing {}.",
        ]
    elif dataset == "CLEVRCount":
        templates = [
            "a photo of {} objects.",
        ]
    elif dataset == "CLEVRDist":
        templates = [
            "a photo where the closest object is {} pixels away.",
        ]
    elif dataset == "ImageNetSketch":
        templates = [
            "a sketch of a {}.",
        ]
    else:
        raise ValueError("Dataset not supported")

    return templates


def standardize_dataset(dataset):
    dataset.prompt_template = dataset_prompt_template(dataset.__class__.__name__)
    print(f"Standardizing dataset for {dataset.__class__.__name__}...")

    if dataset.__class__.__name__ == "Flowers102":
        classes = json.load(open("./data/dataset_classes/flowers102_classes.json", "r"))
        dataset.classes = classes
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        dataset.class_to_idx = class_to_idx

    if dataset.__class__.__name__ == "PCAM":
        dataset.classes = [
            "lymph node",
            "lymph node containing metastatic tumor tissue",
        ]
        dataset.class_to_idx = {
            dataset.classes[i]: i for i in range(len(dataset.classes))
        }

    if dataset.__class__.__name__ == "UCF101":
        classes = json.load(open("./data/dataset_classes/ucf101_classes.json", "r"))
        dataset.classes = classes
        dataset.idx_to_class = {i: classes[i] for i in range(len(classes))}

    assert hasattr(
        dataset, "classes"
    ), f"Dataset {dataset.__class__.__name__} does not have classes"
    assert hasattr(
        dataset, "class_to_idx"
    ), f"Dataset {dataset.__class__.__name__} does not have class_to_idx"

    if hasattr(dataset, "targets"):
        return dataset
    elif hasattr(dataset, "_labels"):
        dataset.targets = dataset._labels
    elif hasattr(dataset, "_samples"):
        sample_length = len(dataset._samples)
        targets = [dataset._samples[i][-1] for i in range(sample_length)]
        dataset.targets = targets
    else:
        targets = [label for image, label in dataset]
        dataset.targets = targets
    return copy.deepcopy(dataset)
