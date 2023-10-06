#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import cv2
import math
import numpy as np
import keras_ocr
from chris_plugin import chris_plugin, PathMapper
from matplotlib import pyplot as plt

__version__ = '0.2.3'

DISPLAY_TITLE = r"""
       _        _                             _            _  _____    _            _   _  __       
      | |      (_)                           | |          | ||_   _|  | |          | | (_)/ _|      
 _ __ | |______ _ _ __ ___   __ _  __ _  ___ | |_ _____  _| |_ | |  __| | ___ _ __ | |_ _| |_ _   _ 
| '_ \| |______| | '_ ` _ \ / _` |/ _` |/ _ \| __/ _ \ \/ / __|| | / _` |/ _ \ '_ \| __| |  _| | | |
| |_) | |      | | | | | | | (_| | (_| |  __/| ||  __/>  <| |__| || (_| |  __/ | | | |_| | | | |_| |
| .__/|_|      |_|_| |_| |_|\__,_|\__, |\___| \__\___/_/\_\\__\___/\__,_|\___|_| |_|\__|_|_|  \__, |
| |                                __/ |  ______                                               __/ |
|_|                               |___/  |______|                                             |___/                                              
"""

parser = ArgumentParser(description='A ChRIS plugin to remove text from images',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument('-f', '--fileFilter', default='png', type=str,
                    help='input file filter(only the extension)')
parser.add_argument('-o', '--outputType', default='png', type=str,
                    help='output file type(only the extension)')
parser.add_argument('-r', '--removeAll', default=False, action="store_true",
                    help='Remove all texts from image using text recognition model')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='My ChRIS plugin',
    category='',  # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',  # supported units: Mi, Gi
    min_cpu_limit='1000m',  # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0  # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    # Typically it's easier to think of programs as operating on individual files
    # rather than directories. The helper functions provided by a ``PathMapper``
    # object make it easy to discover input files and write to output files inside
    # the given paths.
    #
    # Refer to the documentation for more options, examples, and advanced uses e.g.
    # adding a progress bar and parallelism.
    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*.{options.fileFilter}")
    for input_file, output_file in mapper:
        # The code block below is a small and easy example of how to use a ``PathMapper``.
        # It is recommended that you put your functionality in a helper function, so that
        # it is more legible and can be unit tested.

        final_image = inpaint_text(str(input_file), options.removeAll)
        output_file = str(output_file).replace(options.fileFilter, options.outputType)
        print(f"Saving output file as ----->{output_file}<-----\n\n")
        final_image.savefig(output_file)


def inpaint_text(img_path, remove_all):
    pipeline = keras_ocr.pipeline.Pipeline()
    # read image
    print(f"Reading input file from ---->{img_path}<----")
    img = cv2.imread(img_path)

    # Prediction_groups is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    # print image with annotation and boxes
    fig, ax = plt.subplots(nrows=1, figsize=(40, 40))
    image = drawAnnotations(image=img, predictions=prediction_groups[0], ax=ax)

    return image.figure

def drawAnnotations(image, predictions, ax=None):
    """Draw text annotations onto image.

    Args:
        image: The image on which to draw
        predictions: The predictions as provided by `pipeline.recognize`.
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(drawBoxes(image=image, boxes=predictions, thickness=1, boxes_format="boxes"))
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())
    left = []
    right = []
    for word, box in predictions:
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))
    ax.set_yticks([])
    ax.set_xticks([])
    for side, group in zip(["left", "right"], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            ax.annotate(
                text=text,
                xy=xy,
                xytext=(-0.05 if side == "left" else 1.05, y),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "r"},
                color="r",
                fontsize=25,
                horizontalalignment="right" if side == "left" else "left",
            )
    return ax
def drawBoxes(image, boxes, color=(255, 0, 0), thickness=2, boxes_format="boxes"):
    """Draw boxes onto an image.

    Args:
        image: The image on which to draw the boxes.
        boxes: The boxes to draw.
        color: The color for each box.
        thickness: The thickness for each box.
        boxes_format: The format used for providing the boxes. Options are
            "boxes" which indicates an array with shape(N, 4, 2) where N is the
            number of boxes and each box is a list of four points) as provided
            by `keras_ocr.detection.Detector.detect`, "lines" (a list of
            lines where each line itself is a list of (box, character) tuples) as
            provided by `keras_ocr.data_generation.get_image_generator`,
            or "predictions" where boxes is by itself a list of (word, box) tuples
            as provided by `keras_ocr.pipeline.Pipeline.recognize` or
            `keras_ocr.recognition.Recognizer.recognize_from_boxes`.
    """
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == "lines":
        revised_boxes = []
        for line,_ in boxes:
            for box in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == "predictions":
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(
            img=canvas,
            pts=np.int32([box[1]]),
            color=color,
            thickness=thickness,
            isClosed=True,
        )
    return canvas


if __name__ == '__main__':
    main()