"""CCTView Extract

This module contains scripts to run a preconfigured pytorch DenseNet201 CNN
to extract and output features for ReID matching.

Note: The implementation for this module is designed to be run in bulk. It will proccess
all images within SUB DIRECTORIES of the given dataset path.

Source: https://github.com/GeoTrouvetout/Vehicle_ReID

Available functions:

    * run_extractor_on_dataset(dataset_path: str) - returns consolidated features.
    * extract_features_from_dataset(dataset_path: str, emit_to_file: bool) - returns extracted features.
"""
import torch
import torchvision

from cctview import utils

FEATURES_MODEL_PATH = utils.resolve_repo_path("models/VeRI_densenet_ft50.pth")


class DenseNetVeRI(torch.nn.Module):
    """Modified version of the DenseNet201 CNN for Vehicle ReID.

    Source: https://github.com/GeoTrouvetout/Vehicle_ReID
    """

    def __init__(self, old_model, nb_classes):
        super(DenseNetVeRI, self).__init__()
        self.features = old_model.features
        self.mixer = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = torch.nn.Linear(1920, nb_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.mixer(x)
        f = x.view(x.size(0), -1)
        x = self.classifier(f)
        return f, x


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Extend torchvision.datasets.ImageFolder to include the image file
    paths witin the data loader output.

    Source: https://stackoverflow.com/questions/56962318/printing-image-paths-from-the-dataloader-in-pytorch
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def _load_dataset(dataset_path):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = ImageFolderWithPaths(dataset_path, data_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    return dataset, dataloader


def _load_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.densenet201(pretrained=False)
    model = DenseNetVeRI(model, 576)
    if device.type == 'cuda':
        model.load_state_dict(torch.load(
            FEATURES_MODEL_PATH, map_location='cuda'))
    else:
        model.load_state_dict(torch.load(
            FEATURES_MODEL_PATH, map_location='cpu'))
    model = model.to(device)
    model.eval()
    return device, model


def extract_features_from_dataset(dataset_path, emit_to_file=True):
    """Run the feature extractor pipeline on the target dataset.

    Note: This function runs on all images within SUB DIRECTORIES of the dataset path.

    Output will be adjacent to each detected images with the suffix "--features.json".

    Example Features JSON
    ---------------------
    {  
        "id": "253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50",
        "model": "VeRI_densenet_ft50.pth",
        "data": [
            0.000247638818109408,
            0.008388178423047066,
            -0.0002481069532223046,
            0.006181125063449144,
            ...
        ]
    }

    Example Return Dict
    -------------------
    {
        "/dataset/detections/253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50": [
            0.000247638818109408,
            0.008388178423047066,
            -0.0002481069532223046,
            0.006181125063449144,
            ...
        ],
        "/dataset/detections/253-134-S--2020-05-05--13.10.14--detections--2--car--27.4--10x7--51.46v61.53": [
            0.000247638818109408,
            0.008388178423047066,
            -0.0002481069532223046,
            0.006181125063449144,
            ...
        ],
        "/dataset/detections/253-134-S--2020-05-05--13.10.14--detections--3--car--32.1--8x7--64.45v72.52": [
            0.000247638818109408,
            0.008388178423047066,
            -0.0002481069532223046,
            0.006181125063449144,
            ...
        ],
    }

    Parameters
    -----------
    dataset_path: str
        Path to dataset directory containing SUB DIRECTORIES with target image 
        files to be extracted.

    emit_to_file: bool
        If True, output the extracted image features to file. 
        Default: True.

    Returns
    -------
    dict
        Map of features extracted for each file within the dataset.
    """
    dataset, dataloader = _load_dataset(dataset_path)
    print("Extracting features from a dataset of {} images...".format(len(dataset)))
    device, model = _load_torch()
    features = {}
    for img, labels, paths in dataloader:
        img = img.to(device)
        feat, _ = model(img)
        for i, f in enumerate(feat):
            path = paths[i]
            if device.type == 'cuda':
                data = f.cpu().detach().numpy().tolist()
            else:
                data = f.detach().numpy().tolist()
            features[path] = data
            if emit_to_file:
                utils.write_features_to_json_file(
                    path, data, FEATURES_MODEL_PATH)
    return features


def run_extractor_on_dataset(dataset_path):
    """Run the feature extractor pipeline on the target dataset then consolidate results.

    Note: This function runs on all images within SUB DIRECTORIES of the dataset path.

    Output will be a single features JSON for each frame ID. All emitted individual image
    feature files will be removed.

    Example Features JSON
    ---------------------
    {  
        "id": "253-134-S--2020-05-05--13.10.14--features",
        "model": "VeRI_densenet_ft50.pth",
        "253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50": {
            "id": "253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50",
            "data": [
                0.000247638818109408,
                0.008388178423047066,
                -0.0002481069532223046,
                0.006181125063449144,
                ...
            ]
        },
        "253-134-S--2020-05-05--13.10.14--detections--2--car--27.4--10x7--51.46v61.53": {
            "id": "253-134-S--2020-05-05--13.10.14--detections--2--car--27.4--10x7--51.46v61.53",
            "data": [
                0.000247638818109408,
                0.008388178423047066,
                -0.0002481069532223046,
                0.006181125063449144,
                ...
            ]
        }
    }

    Example Return Dict
    ---------------------
    {  
        "253-134-S--2020-05-05--13.10.14": {
            "id": "253-134-S--2020-05-05--13.10.14--features",
            "model": "VeRI_densenet_ft50.pth",
            "253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50": {
                "id": "253-134-S--2020-05-05--13.10.14--detections--1--car--25.3--11x7--113.43v124.50",
                "data": [
                    0.000247638818109408,
                    0.008388178423047066,
                    -0.0002481069532223046,
                    0.006181125063449144,
                    ...
                ]
            },
            "253-134-S--2020-05-05--13.10.14--detections--2--car--27.4--10x7--51.46v61.53": {
                "id": "253-134-S--2020-05-05--13.10.14--detections--2--car--27.4--10x7--51.46v61.53",
                "data": [
                    0.000247638818109408,
                    0.008388178423047066,
                    -0.0002481069532223046,
                    0.006181125063449144,
                    ...
                ]
            }
        }
    }

    Parameters
    -----------
    dataset_path: str
        Path to dataset directory containing SUB DIRECTORIES with target image 
        files to be extracted.

    Returns
    -------
    dict
        Map of features extracted for each frame ID within the dataset.
    """
    extracted_features = extract_features_from_dataset(dataset_path)
    extractor_output = utils.post_process_features_data(
        dataset_path, extracted_features, FEATURES_MODEL_PATH)
    return extractor_output
