import os
import datetime
from pathlib import Path

from cctview import utils
from cctview import detect, extract, match


EXAMPLES_BASE_PATH = utils.resolve_repo_path("examples")

print("\nRunning CCTView Example -- {}".format(datetime.datetime.now()))
print('='*60)

print("Running detector on example images...")
example_images = utils.get_original_images_in_dir(EXAMPLES_BASE_PATH)
print("Found {} original image files in examples directory '{}'".format(
    len(example_images), EXAMPLES_BASE_PATH))
for image in example_images:
    detector_results = detect.run_detector_on_image(
        EXAMPLES_BASE_PATH+"/"+image)
    summary = detector_results.get('summary')
    total = summary.get('total')
    print("Detected {} objects in image '{}'".format(total, image))


example_detections = utils.get_subdirs_in_dir(EXAMPLES_BASE_PATH)
print("Extracting features from {} detection directories in '{}'".format(
    len(example_detections), EXAMPLES_BASE_PATH))
extractor_results = extract.run_extractor_on_dataset(EXAMPLES_BASE_PATH)
print("Output {} features JSON files.".format(len(extractor_results)))


base_frame = '733-681-N'
target_frames = ['735-683-N', '732-680-N', '731-679-N']
matches = match.run_matcher_on_base_and_targets(
    base_frame, target_frames, EXAMPLES_BASE_PATH)
print("Ran matcher on {} objects in base frame '{}'".format(
    len(matches.keys()), base_frame))
