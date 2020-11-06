"""CCTView Detect

This module contains scripts to run a preconfigured ImageAI object detection
model and output the results to file.

Available functions:

    * run_detector_on_image(input_image_file_path: str) - detection results
"""
from imageai.Detection import ObjectDetection

from cctview import utils

DETECTOR_MODEL_PATH = utils.resolve_repo_path(
    'models/resnet50_coco_best_v2.0.1.h5')
DETECTOR = ObjectDetection()
DETECTOR.setModelTypeAsRetinaNet()
DETECTOR.setModelPath(DETECTOR_MODEL_PATH)
DETECTOR.loadModel()
DETECTOR_CUSTOM_OBJECTS = DETECTOR.CustomObjects(
    car=True, motorcycle=True, bus=True, train=True, person=True, bicycle=True)


def run_detector_on_image(input_image_file_path):
    """Run the object detection model on the given image file.

    This function will generate three types of output:
        1. Detections image - a copy of the original input image file
                              with each detected object within their respective
                              bounding boxes.
        2. Detections object directory - a directory containing cropped out 
                            detections from the original input image file
                            for each detected object.
        3. Detections JSON - summary output from running the detection model
                            including each detection type, probability, and
                            bounding boxes.

    Example Detections JSON
    -------------------------
    {
        "id": "735-683-N--2020-05-05--13.34.46",
        "model": "resnet50_coco_best_v2.0.1.h5",
        "summary": {
            "total": 3,
            "car": 3
        },
        "735-683-N--2020-05-05--13.34.46--detections--1--car--83.9--31x31--108.113v139.144": {
            "id": "735-683-N--2020-05-05--13.34.46--detections--1--car--83.9--31x31--108.113v139.144",
            "percentage_probability": 83.87490510940552,
            "type": "car",
            "box_points": [
                108,
                113,
                139,
                144
            ]
        },
        "735-683-N--2020-05-05--13.34.46--detections--2--car--85.4--38x38--169.120v207.158": {
            "id": "735-683-N--2020-05-05--13.34.46--detections--2--car--85.4--38x38--169.120v207.158",
            "percentage_probability": 85.40033102035522,
            "type": "car",
            "box_points": [
                169,
                120,
                207,
                158
            ]
        },
        "735-683-N--2020-05-05--13.34.46--detections--3--car--79.7--57x53--199.148v256.201": {
            "id": "735-683-N--2020-05-05--13.34.46--detections--3--car--79.7--57x53--199.148v256.201",
            "percentage_probability": 79.72716093063354,
            "type": "car",
            "box_points": [
                199,
                148,
                256,
                201
            ]
        }
    }

    Example Return Dict
    -------------------
    {
        "id": "735-683-N--2020-05-05--13.34.46--detections--1--car--83.9--31x31--108.113v139.144",
        "percentage_probability": 83.87490510940552,
        "type": "car",
        "box_points": [
            108,
            113,
            139,
            144
        ]
    }

    Parameters
    -----------
    input_image_file_path: str
        The file to run object detection on.

    Returns
    -------
    dict
        Summary output from running the detection model including each 
        detection type, probability, and bounding boxes.
    """
    output_image_file_path = utils.get_output_detections_image_file_path(
        input_image_file_path)
    output_detections_json_file_path = utils.get_output_detections_json_file_path(
        input_image_file_path)
    detections = DETECTOR.detectCustomObjectsFromImage(
        custom_objects=DETECTOR_CUSTOM_OBJECTS,
        input_image=input_image_file_path,
        output_image_path=output_image_file_path,
        minimum_percentage_probability=40,
        extract_detected_objects=True,
        display_percentage_probability=False,
        display_object_name=False)
    detector_output = utils.write_detections_to_json_file(
        output_detections_json_file_path, detections)
    detector_output = utils.post_process_detections_data(
        input_image_file_path, detections, DETECTOR_MODEL_PATH)
    return detector_output
