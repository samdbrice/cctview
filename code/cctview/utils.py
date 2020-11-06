"""CCTView Utils

This module contains various utility functions for reading and writing files.

Available functions:

    * resolve_repo_path(repo_item_path: str) - returns absolute path to an item within the directory.
"""
import os
import json
from pathlib import Path
import shutil


def post_process_features_data(dataset_path, extractions, features_model_path):
    """Post-process features data.

    This function does two things:
        1. Outputs a consilidated "--features.json" for each frame.
        2. Deletes any individualy emitted "--features.json" for dataset images.

    It returns a maps of keys and value for each consolidated frame extracted
    features

    Parameters
    -----------
    dataset_path: str
        Path to dataset directory containing SUB DIRECTORIES with target image 
        files to be extracted.

    extractions: dict
        Map of features extracte dfrom the dataset.
        See `extract.extract_features_from_dataset` for details.

    features_model_path: str
        Absolute path to model used to extract the features.

    Returns
    -------
    dict
        Map of features extracted for each frame ID within the dataset.
        See `extract.run_extractor_on_dataset` for details.
    """
    features_json_files = {}
    for path, features in extractions.items():
        frame_id = get_frame_id(path)
        if features_json_files.get(frame_id) is None:
            features_json_files[frame_id] = {
                "id": frame_id+"--features",
                "model": features_model_path.split('/')[-1]
            }
        object_id = get_object_id(path)
        object_features = {
            "id": object_id,
            "data": features
        }
        features_json_files.get(frame_id)[object_id] = object_features

    for frame_id, data in features_json_files.items():
        features_json_file_path = dataset_path+"/"+frame_id+"--features.json"
        with open(features_json_file_path, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=2)

    for path in extractions.keys():
        remove_features_json_file(path)

    return features_json_files


def post_process_detections_data(input_file_path, detections, detector_model_path):
    """Post-process detections data.

    This function does two things:
        1. Overrides the default ImageAI detections JSON with something more legible.
        2. Rewrites and renames the detected objects directory and contents to something more legible.

    It returns the restructured ImageAI detections JSON.

    Parameters
    -----------
    input_image_file_path: str
        The file object detection was run on.

    detections: dict
        Detections dict returned from ImageAI.

    detector_model_path: str
        Absolute path to model used in object detection.

    Returns
    -------
    dict
        Summary output from running the detection model including each 
        detection type, probability, and bounding boxes.
        See `detect.run_detector_on_image` for details.
    """
    frame_id = get_object_id(input_file_path)
    detections_metadata = detections[0]
    detections_files = detections[1]
    detections_data = {
        "id": frame_id,
        "model": detector_model_path.split('/')[-1],
        "summary": {
            "total": len(detections_files)
        }
    }
    for ix, detection_metadata in enumerate(detections_metadata):
        name = detection_metadata.get('name')
        percentage_probability = detection_metadata.get(
            'percentage_probability')
        box_points = detection_metadata.get('box_points')
        prob = round(percentage_probability, 1)
        x1, y1, x2, y2 = box_points
        width = (x2-x1)
        height = (y2-y1)
        detection_ids = [
            frame_id,
            "detections",
            str(ix+1),
            name,
            str(prob),
            "{}x{}".format(width, height),
            "{}.{}v{}.{}".format(x1, y1, x2, y2)
        ]
        detection_id = "--".join(detection_ids)
        old_file_path = detections_files[ix]
        old_file_paths = old_file_path.split('/')
        old_file_paths[-1] = detection_id+".jpeg"
        new_file_path = '/'.join(old_file_paths)
        os.rename(old_file_path, new_file_path)
        summary = detections_data.get('summary')
        if summary.get(name) is None:
            summary[name] = 1
        else:
            summary[name] += 1
        detections_data[detection_id] = {
            "id": detection_id,
            "type": name,
            "percentage_probability": percentage_probability,
            "box_points": box_points
        }
    output_image_file_path = get_output_detections_image_file_path(
        input_file_path)
    old_objects_dir_path = output_image_file_path+"-objects"
    new_objects_dir_path = '.'.join(old_objects_dir_path.split('.')[0:-1])
    shutil.rmtree(new_objects_dir_path, ignore_errors=True)
    os.rename(old_objects_dir_path, new_objects_dir_path)
    output_detections_json_file_path = get_output_detections_json_file_path(
        input_file_path)
    write_detections_to_json_file(
        output_detections_json_file_path, detections_data)
    return detections_data


def post_process_matches_data(matches, base_frame, target_frames, base_path):
    """Post-process matches data.

    This function output restults from `match.run_matcher_on_base_and_targets` 
    into a directory `matches` at the top-level of the repository.

    Highly customized, not reusable. 
    
    See `match.run_matcher_on_base_and_targets` for details.
    """
    with open(base_path+"/"+base_frame+"--matches.json", 'w') as outfile:
        json.dump(matches, outfile, sort_keys=True, indent=2)
    for base_detection, target_matches_map in matches.items():
        matches_dir_path = base_path+"/../matches/"+base_detection
        shutil.rmtree(matches_dir_path, ignore_errors=True)
        os.makedirs(matches_dir_path)
        frame_id = get_frame_id(base_detection)
        from_file = base_path+"/"+frame_id+"--detections/"+base_detection+".jpeg"
        to_file = matches_dir_path+"/base--"+base_detection+".jpeg"
        shutil.copyfile(from_file, to_file)
        for ix, target_frame in enumerate(target_frames):
            target_matches = target_matches_map.get(target_frame)
            for object_id, distance in target_matches.items():
                frame_id = get_frame_id(object_id)
                from_file = base_path+"/"+frame_id+"--detections/"+object_id+".jpeg"
                to_file = matches_dir_path+"/cam-" + \
                    str(ix+1)+"--"+str(distance)+"--"+base_detection+".jpeg"
                shutil.copyfile(from_file, to_file)
    return matches


def resolve_repo_path(repo_item_path):
    """Resolve the absolute path for an item within the repository.

    Parameters
    -----------
    repo_abs_path: str
        Path of an item from within the repo.

    Returns
    -------
    str
        Fully resolved path to the target item.
    """
    path = Path(os.path.join(
        os.path.dirname(__file__), "../..", repo_item_path)).resolve()
    return str(path)


def get_frame_id(input_image_path):
    """Get the frame ID based on the file name.

    Returns the composite frame ID based on:
        - Cam ID
        - Date
        - Time

    Parameters
    -----------
    input_image_path: str
        Path of a given frame or image.

    Returns
    -------
    str
        Composite ID based on Cam, Date, and Time.
    """
    input_image_path = input_image_path.replace('--original', '')
    file_name = input_image_path.split('/')[-1]
    frame_id = "--".join(file_name.split("--")[0:3])
    return frame_id


def get_object_id(input_image_path):
    """Get the object ID based on the file name.

    Returns the composite ID based on frame ID and
    object specific metadata. 

    Effectively removes path and file extension info.

    Parameters
    -----------
    input_image_path: str
        Path of a given object image.

    Returns
    -------
    str
        Composite ID based on frame ID and object specific metadata. 
    """
    input_image_path = input_image_path.replace('--original', '')
    file_name = input_image_path.split('/')[-1]
    object_id = '.'.join(file_name.split('/')[-1].split('.')[:-1])
    return object_id


def get_original_images_in_dir(dir_path):
    """Get images within the target dir suffixed with "--original".
    
    Strictly for the examples script.

    Parameters
    -----------
    dir_path: str
        Path to examples directory.

    Returns
    -------
    list of str
        Original images to run the example script on.
    """
    if not os.path.exists(dir_path):
        return []
    onlyfiles = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and "--original" in f
    ]
    return onlyfiles


def get_features_json_files_in_dir(dir_path):
    """Get files within the target dir suffixed with "--features.json".
    
    Strictly for the examples script.

    Parameters
    -----------
    dir_path: str
        Path to examples directory.

    Returns
    -------
    list of str
        Features JSON files.
    """
    if not os.path.exists(dir_path):
        return []
    onlyfiles = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and "--features.json" in f
    ]
    return onlyfiles


def get_subdirs_in_dir(dir_path):
    """Get first-level subdirs within the given directory.
    
    Strictly for the examples script.

    Parameters
    -----------
    dir_path: str
        Path to examples directory.

    Returns
    -------
    list of str
        Sub directories.
    """
    if not os.path.exists(dir_path):
        return []
    onlydirs = [
        f for f in os.listdir(dir_path)
        if not os.path.isfile(os.path.join(dir_path, f))
    ]
    return onlydirs


def get_output_detections_image_file_path(input_file_path, suffix="--detections"):
    """Get the appropriate output image path for a given image input.

    Effectively appends "--detections" to the original image file and 
    places it within the same directory.

    Parameters
    -----------
    input_file_path: str
        Path to input image.

    suffix: str
        Suffix appended to the file.
        Default: "--detections"

    Returns
    -------
    str
        Full path for detections output image.
    """
    input_file_path = input_file_path.replace('--original.', '.')
    input_file_paths = input_file_path.split('.')
    input_file_paths[-2] = input_file_paths[-2]+suffix
    return '.'.join(input_file_paths)


def get_output_detections_json_file_path(input_file_path, suffix="--detections"):
    """Get the appropriate detections JSON output path for a given image input.

    Effectively appends "--detections" to the original image file and 
    places it within the same directory.

    Parameters
    -----------
    input_file_path: str
        Path to input image.

    suffix: str
        Suffix appended to the file.
        Default: "--detections"

    Returns
    -------
    str
        Full path for detections JSON output.
    """
    input_file_path = input_file_path.replace('--original.', '.')
    input_file_paths = input_file_path.split('.')
    input_file_paths[-2] = input_file_paths[-2]+suffix
    input_file_paths[-1] = 'json'
    return '.'.join(input_file_paths)


def get_features_json_file_path(input_image_path):
    """Get the appropriate features JSON output path for a given image input.

    Effectively appends "--features" to the original image file and 
    places it within the same directory.

    Parameters
    -----------
    input_file_path: str
        Path to input image.
        
    Returns
    -------
    str
        Full path for features JSON output.
    """
    features_file_paths = input_image_path.split('.')
    features_file_paths[-2] += '--features'
    features_file_paths[-1] = 'json'
    features_file_path = '.'.join(features_file_paths)
    return features_file_path


def get_first_item_that_startswith(items, starts_with):
    """Get the first item within the list that starts with a specific string.

    Parameters
    -----------
    items: list of str
        Path to input image.

    starts_with: str
        String to search for.
        
    Returns
    -------
    str
        First item in the list that starts with the given string.
    """
    starters = [item for item in items if item.startswith(starts_with)]
    return starters[0]


def write_detections_to_json_file(file_path, detections):
    """Output detections to JSON file.

    Parameters
    -----------
    file_path: str
        Path to output file.

    detections: dict
        Detections data.
        
    Returns
    -------
    dict
        Detections data.
    """
    with open(file_path, 'w') as outfile:
        json.dump(detections, outfile, sort_keys=True, indent=2)
    return detections


def write_features_to_json_file(input_image_path, data, features_model_path):
    """Output features of given input image to JSON file.

    Parameters
    -----------
    input_image_path: str
        Path to original input image.

    data: list of float
        Features data.

    features_model_path: str
        Absolute path to model used to extract the features.
        
    Returns
    -------
    dict
        Features JSON output.
    """
    object_id = get_object_id(input_image_path)
    features_file_path = get_features_json_file_path(input_image_path)
    features_data = {
        "id": object_id,
        "model": features_model_path.split('/')[-1],
        "data": data
    }
    with open(features_file_path, 'w') as outfile:
        json.dump(features_data, outfile, sort_keys=True, indent=2)
    return features_data


def remove_features_json_file(input_image_path):
    """Remove any existing features output for the given image input.

    Parameters
    -----------
    input_image_path: str
        Path to original input image.
        
    Returns
    -------
    bool
        True if a file was removed. False, otherwise.
    """
    features_file_path = get_features_json_file_path(input_image_path)
    try:
        os.remove(features_file_path)
        return True
    except:
        pass
    return False


def load_json_file(file_path):
    """Load the given JSON path.

    Parameters
    -----------
    file_path: str
        Path to target JSON file.
        
    Returns
    -------
    dict
        Loaded JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def load_features_json_file(base_path, features_file_path):
    """Load the given features JSON file.

    Parameters
    -----------
    base_path: str
        Base path for all features files.

    features_file_path: str
        Relative path to target features JSON file.
        
    Returns
    -------
    dict
        Loaded features JSON file.
    """
    return load_json_file(base_path+"/"+features_file_path)
