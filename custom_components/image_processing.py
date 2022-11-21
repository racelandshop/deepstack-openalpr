"""
Component that will perform object detection and identification via deepstack.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.deepstack_object
"""
from collections import namedtuple, Counter
import datetime
import io
import logging
import os
import re
from datetime import timedelta
from typing import Tuple, Dict, List
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw
import requests

import deepstack.core as ds
import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.util.pil import draw_box
from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    DEFAULT_CONFIDENCE,
    DOMAIN,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_NAME,
    CONF_IP_ADDRESS,
    CONF_PORT,
)
from homeassistant.core import split_entity_id

_LOGGER = logging.getLogger(__name__)

ANIMAL = "animal"
ANIMALS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]
OTHER = "other"
PERSON = "person"
LICENSE_PLATE = "licence-plate"
VEHICLE = "vehicle"
VEHICLES = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
OBJECT_TYPES = [ANIMAL, OTHER, PERSON, VEHICLE]


CONF_API_KEY = "api_key"
CONF_TARGET = "target"
CONF_TARGETS = "targets"
CONF_TIMEOUT = "timeout"
CONF_SAVE_FILE_FORMAT = "save_file_format"
CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_SAVE_TIMESTAMPTED_FILE = "save_timestamped_file"
CONF_ALWAYS_SAVE_LATEST_FILE = "always_save_latest_file"
CONF_SHOW_BOXES = "show_boxes"
CONF_SCALE = "scale"
CONF_CUSTOM_MODEL = "custom_model"
CONF_CROP_ROI = "crop_to_roi"

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
DEFAULT_API_KEY = ""
DEFAULT_TARGETS = [{CONF_TARGET: PERSON}]
DEFAULT_TIMEOUT = 10
DEAULT_SCALE = 1.0

EVENT_OBJECT_DETECTED = "deepstack.object_detected"
BOX = "box"
FILE = "file"
OBJECT = "object"
SAVED_FILE = "saved_file"
MIN_CONFIDENCE = 0.1
JPG = "jpg"
PNG = "png"

# rgb(red, green, blue)
RED = (255, 0, 0)  # For objects within the ROI
GREEN = (0, 255, 0)  # For ROI box
YELLOW = (255, 255, 0)  # Unused

TARGETS_SCHEMA = {
    vol.Required(CONF_TARGET): cv.string,
    vol.Optional(CONF_CONFIDENCE): vol.All(
        vol.Coerce(float), vol.Range(min=10, max=100)
    ),
}


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_IP_ADDRESS): cv.string,
        vol.Required(CONF_PORT): cv.port,
        vol.Optional(CONF_API_KEY, default=DEFAULT_API_KEY): cv.string,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_CUSTOM_MODEL, default=""): cv.string,
        vol.Optional(CONF_TARGETS, default=DEFAULT_TARGETS): vol.All(
            cv.ensure_list, [vol.Schema(TARGETS_SCHEMA)]
        ),
        vol.Optional(CONF_SCALE, default=DEAULT_SCALE): vol.All(
            vol.Coerce(float, vol.Range(min=0.1, max=1))
        ),
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
        vol.Optional(CONF_SAVE_FILE_FORMAT, default=JPG): vol.In([JPG, PNG]),
        vol.Optional(CONF_SAVE_TIMESTAMPTED_FILE, default=False): cv.boolean,
        vol.Optional(CONF_ALWAYS_SAVE_LATEST_FILE, default=False): cv.boolean,
        vol.Optional(CONF_SHOW_BOXES, default=True): cv.boolean,
        vol.Optional(CONF_CROP_ROI, default=False): cv.boolean,
    }
)

Box = namedtuple("Box", "y_min x_min y_max x_max")
Point = namedtuple("Point", "y x")


def point_in_box(box: Box, point: Point) -> bool:
    """Return true if point lies in box"""
    if (box.x_min <= point.x <= box.x_max) and (box.y_min <= point.y <= box.y_max):
        return True
    return False


def object_in_roi(roi: dict, centroid: dict) -> bool:
    """Convenience to convert dicts to the Point and Box."""
    target_center_point = Point(centroid["y"], centroid["x"])
    roi_box = Box(roi["y_min"], roi["x_min"], roi["y_max"], roi["x_max"])
    return point_in_box(roi_box, target_center_point)


def get_valid_filename(name: str) -> str:
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def get_object_type(object_name: str) -> str:
    if object_name == PERSON:
        return PERSON
    elif object_name == LICENSE_PLATE:
        return LICENSE_PLATE
    elif object_name in ANIMALS:
        return ANIMAL
    elif object_name in VEHICLES:
        return VEHICLE
    else:
        return OTHER


def get_objects(predictions: list, img_width: int, img_height: int) -> List[Dict]:
    """Return objects with formatting and extra info."""
    objects = []
    decimal_places = 3
    for pred in predictions:
        box_width = pred["x_max"] - pred["x_min"]
        box_height = pred["y_max"] - pred["y_min"]
        box = {
            "height": round(box_height / img_height, decimal_places),
            "width": round(box_width / img_width, decimal_places),
            "y_min": round(pred["y_min"] / img_height, decimal_places),
            "x_min": round(pred["x_min"] / img_width, decimal_places),
            "y_max": round(pred["y_max"] / img_height, decimal_places),
            "x_max": round(pred["x_max"] / img_width, decimal_places),
        }
        box_area = round(box["height"] * box["width"], decimal_places)
        centroid = {
            "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
            "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
        }
        name = pred["label"]
        object_type = get_object_type(name)
        confidence = round(pred["confidence"] * 100, decimal_places)

        objects.append(
            {
                "bounding_box": box,
                "box_area": box_area,
                "centroid": centroid,
                "name": name,
                "object_type": object_type,
                "confidence": confidence,
            }
        )
    return objects


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up the classifier."""
    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    if save_file_folder:
        save_file_folder = Path(save_file_folder)

    entities = []
    for camera in config[CONF_SOURCE]:
        object_entity = ObjectClassifyEntity(
            ip_address=config.get(CONF_IP_ADDRESS),
            port=config.get(CONF_PORT),
            api_key=config.get(CONF_API_KEY),
            timeout=config.get(CONF_TIMEOUT),
            custom_model=config.get(CONF_CUSTOM_MODEL),
            targets=config.get(CONF_TARGETS),
            confidence=config.get(CONF_CONFIDENCE),
            scale=config[CONF_SCALE],
            show_boxes=config[CONF_SHOW_BOXES],
            save_file_folder=save_file_folder,
            save_file_format=config[CONF_SAVE_FILE_FORMAT],
            save_timestamped_file=config.get(CONF_SAVE_TIMESTAMPTED_FILE),
            always_save_latest_file=config.get(CONF_ALWAYS_SAVE_LATEST_FILE),
            crop_roi=config[CONF_CROP_ROI],
            camera_entity=camera.get(CONF_ENTITY_ID),
            name=camera.get(CONF_NAME),
        )
        entities.append(object_entity)
    add_devices(entities)


class ObjectClassifyEntity(ImageProcessingEntity):
    """Perform a object classification."""

    def __init__(
        self,
        ip_address,
        port,
        api_key,
        timeout,
        custom_model,
        targets,
        confidence,
        scale,
        show_boxes,
        save_file_folder,
        save_file_format,
        save_timestamped_file,
        always_save_latest_file,
        crop_roi,
        camera_entity,
        name=None,
    ):
        """Init with the API key and model id."""
        super().__init__()
        self._dsobject = ds.DeepstackObject(
            ip=ip_address,
            port=port,
            api_key=api_key,
            timeout=timeout,
            min_confidence=MIN_CONFIDENCE,
            custom_model=custom_model,
        )
        self._custom_model = custom_model
        self._confidence = confidence
        self._summary = {}
        self._targets = targets
        for target in self._targets:
            if CONF_CONFIDENCE not in target.keys():
                target.update({CONF_CONFIDENCE: self._confidence})
        self._targets_names = [
            target[CONF_TARGET] for target in targets
        ]  # can be a name or a type
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            camera_name = split_entity_id(camera_entity)[1]
            self._name = "deepstack_object_{}".format(camera_name)

        self._state = None
        self._objects = []  # The parsed raw data
        self._targets_found = []
        self._last_detection = None

        self._roi_dict = []
        self._crop_roi = True
        self._scale = scale
        self._show_boxes = show_boxes
        self._image_width = None
        self._image_height = None
        self._save_file_folder = save_file_folder
        self._save_file_format = save_file_format
        self._always_save_latest_file = always_save_latest_file
        self._save_timestamped_file = save_timestamped_file
        self._always_save_latest_file = always_save_latest_file
        self._image = []
        self._croped_image = None

    def process_image(self, image):
        """Process an image."""
        image_data = open('/config/www/deepstack/plates/vugo.jpg',"rb").read()
        self._image.append(Image.open(io.BytesIO(bytearray(image_data))))
        self._image_width, self._image_height = self._image[0].size
        self._state = None
        self._objects = []  # The parsed raw data
        self._targets_found = []
        self._summary = {}
        saved_image_path = None
        
        try:
            predictions = self._dsobject.detect(image_data)
        except ds.DeepstackException as exc:
            _LOGGER.error("Deepstack error : %s", exc)
            return

        self._objects = get_objects(predictions, self._image_width, self._image_height)

        for target in self._targets:
            if target['target'] == 'car':
                confidence_car = target['confidence']

        new_obj = []
        
        for i in range(len(self._objects)):
            if self._objects[i]['name'] == 'car' or self._objects[i]['name'] == 'truck' or self._objects[i]['name'] == 'motorcycle' or self._objects[i]['name'] == 'bicycle' or self._objects[i]['name'] == 'bus':
                new_obj.append(self._objects[i])

        car_obj = []
        for i in range(len(new_obj)):
            if new_obj[i]['confidence'] > confidence_car:
                car_obj.append(new_obj[i])

        if (len(car_obj) > 0):
            for i in range(len(car_obj)):
                if (car_obj[i]['confidence'] > confidence_car):
                    self._roi_dict.append({
                        "y_min": car_obj[i]['bounding_box']["y_min"],
                        "x_min": car_obj[i]['bounding_box']["x_min"],
                        "y_max": car_obj[i]['bounding_box']["y_max"],
                        "x_max": car_obj[i]['bounding_box']["x_max"],
                    })
                    # scale to roi
                    if self._crop_roi:
                        roi = (
                            self._image_width * self._roi_dict[i]["x_min"],
                            self._image_height * self._roi_dict[i]["y_min"],
                            self._image_width * self._roi_dict[i]["x_max"],
                            self._image_height * self._roi_dict[i]["y_max"],
                        )
                        self._image.append(self._image[0].crop(roi))
                    if self._scale != DEAULT_SCALE:
                        newsize = (self._image_width * self._scale, self._image_width * self._scale)
                        self._image[i].thumbnail(newsize, Image.ANTIALIAS)
            del self._image[0]
            for i in range(len(self._image)):
                with io.BytesIO() as output:
                    self._image[i].save(output, format="JPEG")
                    image = output.getvalue()
                _LOGGER.debug(
                    (
                        f"Image cropped with : {self._roi_dict} W={self._image_width} H={self._image_height}"
                    )
                )

        self._targets_found = []

        for obj in self._objects:
            if not (
                (obj["name"] in self._targets_names)
                or (obj["object_type"] in self._targets_names)
            ):
                continue
            ## Then check if the type has a configured confidence, if yes assign
            ## Then if a confidence for a named object, this takes precedence over type confidence
            confidence = None
            open_alpr_fishing_target = []
            for target in self._targets:
                if obj["object_type"] == target[CONF_TARGET]:
                    confidence = target[CONF_CONFIDENCE]
            for target in self._targets:
                if obj["name"] == target[CONF_TARGET]:
                    confidence = target[CONF_CONFIDENCE]
            if obj["confidence"] > confidence:
                if not self._crop_roi and not object_in_roi(self._roi_dict, obj["centroid"]):
                    continue
                self._targets_found.append(obj)
            if (obj["confidence"] < confidence) and (obj["confidence"] > 30):
                open_alpr_fishing_target.append(obj)
    
        print('found', self._targets_found)

        self._state = len(self._targets_found)
        if self._state > 0:
            self._last_detection = dt_util.now().strftime(DATETIME_FORMAT)

        if self._save_file_folder:
            if self._state > 0 or self._always_save_latest_file:
                saved_image_path = self.save_image(
                    # self._targets_found,
                    self._save_file_folder,
                )

        if (len(self._targets_found) > 0):

            # for target in self._targets_found:

            #     if target['name'] == "car" or target['name'] == 'truck' or target['name'] == 'motorcycle' or target['name'] == 'bicycle' or target['name'] == 'bus':
                    for i in range(len(saved_image_path)):

                        try:
                            #/config/www/deepstack/deepstack_object_hd_ipc_profile_000_latest.jpg
                            image_data = open(saved_image_path[i],"rb").read()
                            payload = {'country_code': 'eu'}
                            response = requests.post("http://192.168.1.142:3000/detect",files={"upload":image_data}, data=payload).json()
                            result = response['results']
                            plate = {}
                            if len(result) != 0:
                                for i in range(len(result)):
                                    plate['name'] = result[i]['plate']
                                    plate['object_type'] = 'licence-plate'
                                    plate['confidence'] = result[i]['confidence']
                                    self._targets_found.append(plate)
                                    print("PLATEeeeeeeEEE", plate)

                        except Exception as e:
                            _LOGGER.error("error getting text:", e)
                            return
        elif (len(open_alpr_fishing_target) > 0 and len(self._targets_found) == 0):
            try:
                # for target in open_alpr_fishing_target:

                #     if target['name'] == "car" or target['name'] == 'truck' or target['name'] == 'motorcycle' or target['name'] == 'bicycle' or target['name'] == 'bus':
                        for i in range(len(saved_image_path)):
                            try:
                                print('path', saved_image_path[i])
                                with io.BytesIO() as output:
                                    #/config/www/deepstack/deepstack_object_hd_ipc_profile_000_latest.jpg
                                    image_data = open(saved_image_path[i],"rb").read()
                                    payload = {'country_code': 'eu'}
                                    response = requests.post("http://192.168.1.142:3000/detect",files={"upload":image_data}, data=payload).json()
                                    result = response['results']
                                    plate = {}
                                    if len(result) != 0:
                                        for i in range(len(result)):
                                            plate['name'] = result[i]['plate']
                                            plate['object_type'] = 'licence-plate'
                                            plate['confidence'] = result[i]['confidence']
                                            self._targets_found.append(plate)
                                    print("PLATEeeeeeeEEE", plate)
                            except Exception as e:
                                _LOGGER.error("error getting plate:", e)
                                return

            except Exception as e:
                _LOGGER.error("Not able to get any fishing plates:", e)
                return
        
        targets_found = [
            obj["name"] for obj in self._targets_found
        ]  # Just the list of target names, e.g. [car, car, person]
        self._summary = dict(Counter(targets_found))  # e.g. {'car':2, 'person':1}


        # Fire events
        for target in self._targets_found:
            target_event_data = target.copy()
            target_event_data[ATTR_ENTITY_ID] = self.entity_id
            if len(saved_image_path) > 0:
                for i in range(len(saved_image_path)):
                    target_event_data[SAVED_FILE] = saved_image_path[i]
                    self.hass.bus.fire(EVENT_OBJECT_DETECTED, target_event_data)

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        return "targets"

    @property
    def should_poll(self):
        """Return the polling state."""
        return False

    @property
    def extra_state_attributes(self) -> Dict:
        """Return device specific state attributes."""
        attr = {}
        attr["targets"] = self._targets
        attr["targets_found"] = [
            {obj["name"]: obj["confidence"]} for obj in self._targets_found
        ]
        attr["summary"] = self._summary
        if self._last_detection:
            attr["last_target_detection"] = self._last_detection
        if self._custom_model:
            attr["custom_model"] = self._custom_model
        attr["all_objects"] = [
            {obj["name"]: obj["confidence"]} for obj in self._objects
        ]
        if self._save_file_folder:
            attr[CONF_SAVE_FILE_FOLDER] = str(self._save_file_folder)
            attr[CONF_SAVE_FILE_FORMAT] = self._save_file_format
            attr[CONF_SAVE_TIMESTAMPTED_FILE] = self._save_timestamped_file
            attr[CONF_ALWAYS_SAVE_LATEST_FILE] = self._always_save_latest_file
        return attr

    def save_image(self, directory) -> list:
        """Draws the actual bounding box of the detected objects.

        Returns: saved_image_path, which is the path to the saved timestamped file if configured, else the default saved image.
        """
        saved_image_path = [None] * len(self._image)
        for i in range(len(self._image)):
            try:
                img = self._image[i].convert("RGB")
            except UnidentifiedImageError:
                _LOGGER.warning("Deepstack unable to process image, bad data")
                return
            
            # Save images, returning the path of saved image as str
            latest_save_path = (
                directory
                / f"{get_valid_filename(self._name).lower()}_latest{i}.{self._save_file_format}"
            )
            img.save(latest_save_path)
            _LOGGER.info("Deepstack saved file %s", latest_save_path)
            saved_image_path[i] = str(latest_save_path)

        return saved_image_path
