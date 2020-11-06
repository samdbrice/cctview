"""CCTView Match

This module contains scripts to run a matcher for ReID.

Available functions:

    * calc_med(p_data: list of float, q_data: list of float) - returns Minimal Euclidean Distance.
    * run_matcher_on_base_and_targets(base_frame: str, target_frames: str, base_path: str) - returns matches data.
"""
import torch

from cctview import utils


def calc_med(p_data, q_data):
    """Calculate the Minimal Euclidean Distance between two points.

    Parameters
    -----------
    p_data: list of float
        First  point.

    q_data: list of float
        Second  point.

    Returns
    -------
    float
        Minimal Euclidean Distance between two points.
    """
    p = torch.FloatTensor(p_data)
    q = torch.FloatTensor(q_data)
    return torch.norm(p - q).item()


def run_matcher_on_base_and_targets(base_frame, target_frames, base_path):
    """Run a matcher script to determine MED from base frame.

    Highly customized matcher script. Generates two types of output:
        1. Matches JSON - nested matches from base to targets
        2. Matches directory - directory for each base object with 
                                target matches within.

    For each object in base frame, calculates MED for each object
    in target frames, then writes outout to JSON.

    Matches directory for each base frame object contains the base object
    image file along with each target object ad their respective distance.
    Images have been renamed for easier sortability.

    Example Matches Directory
    -------------------------
    matches
    ├── 733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65
    │   ├── base--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-1--37.29322052001953--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-1--37.588375091552734--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-1--40.333370208740234--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--23.661474227905273--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--24.81173324584961--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--25.880002975463867--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--27.87835693359375--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--29.547557830810547--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-2--29.633991241455078--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--17.623247146606445--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--22.22691535949707--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--34.63451385498047--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--35.772640228271484--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--37.2514533996582--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   ├── cam-3--38.56435775756836--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    │   └── cam-3--39.74309158325195--733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65.jpeg
    ├── 733-681-N--2020-05-05--13.34.29--detections--2--car--43.5--12x10--154.56v166.66
    │   ├── base--733-681-N--2020-05-05--13.34.29--detections--2--car--43.5--12x10--154.56v166.66.jpeg
    │   ├── cam-1--40.17823791503906--733-681-N--2020-05-05--13.34.29--detections--2--car--43.5--12x10--154.56v166.66.jpeg
    │   ├── cam-1--40.98042678833008--733-681-N--2020-05-05--13.34.29--detections--2--car--43.5--12x10--154.56v166.66.jpeg
    │   ├── cam-1--42.724021911621094--733-681-N--2020-05-05--13.34.29--detections--2--car--43.5--12x10--154.56v166.66.jpeg
    │   ├── cam-2--25.171867
    ...


    Example Matches JSON
    --------------------
    {
        "733-681-N--2020-05-05--13.34.29--detections--1--car--55.9--10x11--88.54v98.65": {
            "731-679-N": {
            "731-679-N--2020-05-05--13.36.56--detections--1--car--78.0--16x10--84.86v100.96": 35.772640228271484,
            "731-679-N--2020-05-05--13.36.56--detections--2--car--77.2--23x12--176.92v199.104": 22.22691535949707,
            "731-679-N--2020-05-05--13.36.56--detections--3--car--55.0--25x12--221.98v246.110": 17.623247146606445,
            "731-679-N--2020-05-05--13.36.56--detections--4--car--76.0--26x18--111.105v137.123": 34.63451385498047,
            "731-679-N--2020-05-05--13.36.56--detections--5--car--48.7--29x32--56.114v85.146": 38.56435775756836,
            "731-679-N--2020-05-05--13.36.56--detections--6--car--95.3--57x35--206.128v263.163": 39.74309158325195,
            "731-679-N--2020-05-05--13.36.56--detections--7--car--96.4--53x52--82.152v135.204": 37.2514533996582
            },
            "732-680-N": {
            "732-680-N--2020-05-05--13.36.52--detections--1--car--44.5--11x8--57.67v68.75": 29.633991241455078,
            "732-680-N--2020-05-05--13.36.52--detections--2--car--52.5--10x7--133.72v143.79": 24.81173324584961,
            "732-680-N--2020-05-05--13.36.52--detections--3--car--70.0--15x16--2.79v17.95": 27.87835693359375,
        ...
    }


    Parameters
    -----------
    base_frame: str
        Base frame and base objects.

    target_frames: list of str
        Target frames in order with respective target objects.

    base_path: str
        Base path containing base and target frame directories.

    Returns
    -------
    dict
        Matches JSON.
    """
    print("Running matcher for all objects in frame '{}' with frames: {}".format(
        base_frame, ", ".join(target_frames)))
    features_json_files = utils.get_features_json_files_in_dir(base_path)
    base_features_file_path = utils.get_first_item_that_startswith(
        features_json_files, base_frame)

    print('Loading base features file: {}'.format(base_features_file_path))
    base_features = utils.load_features_json_file(
        base_path, base_features_file_path)
    base_detections = [key for key in base_features.keys()
                       if "--detections--" in key]
    base_detections.sort()

    print('Running matcher on {} base detections:'.format(len(base_detections)))
    print("  "+"\n  ".join(base_detections))

    matches = {}

    for target_frame in target_frames:
        print("Matching on target frame '{}'...".format(target_frame))
        target_features_file_path = utils.get_first_item_that_startswith(
            features_json_files, target_frame)
        target_frame_features = utils.load_features_json_file(
            base_path, target_features_file_path)

        target_frame_detections = [
            key for key in target_frame_features.keys() if "--detections--" in key]
        target_frame_detections.sort()

        for base_detection in base_detections:

            if matches.get(base_detection) is None:
                matches[base_detection] = {}
            if matches[base_detection].get(target_frame) is None:
                matches[base_detection][target_frame] = {}

            base_data = base_features.get(base_detection).get('data')
            print("  {}".format(base_detection))

            for target_frame_detection in target_frame_detections:
                target_data = target_frame_features.get(
                    target_frame_detection).get('data')

                distance = calc_med(base_data, target_data)

                matches[base_detection][target_frame][target_frame_detection] = distance

                print("    {} --> {}".format(round(distance, 2),
                                             target_frame_detection))

    matches = utils.post_process_matches_data(
        matches, base_frame, target_frames, base_path)

    return matches
