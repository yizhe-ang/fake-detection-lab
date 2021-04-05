from importlib import import_module
from typing import Dict, List

import yaml


# def read_image(image_uri: Union[Path, str], grayscale=False) -> np.array:
#     """Read image_uri."""

#     def read_image_from_filename(image_filename, imread_flag):
#         # FIXME Change order of channels
#         return cv2.imread(str(image_filename), imread_flag)

#     def read_image_from_url(image_url, imread_flag):
#         url_response = urlopen(str(image_url))  # nosec
#         img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
#         return cv2.imdecode(img_array, imread_flag)

#     imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
#     local_file = os.path.exists(image_uri)
#     try:
#         img = None
#         if local_file:
#             img = read_image_from_filename(image_uri, imread_flag)
#         else:
#             img = read_image_from_url(image_uri, imread_flag)
#         assert img is not None
#     except Exception as e:
#         raise ValueError("Could not load image at {}: {}".format(image_uri, e))
#     return img


# def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
#     """Write image to file."""
#     cv2.imwrite(str(filename), image)


# class TqdmUpTo(tqdm):
#     """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

#     def update_to(self, blocks=1, bsize=1, tsize=None):
#         """
#         Parameters
#         ----------
#         blocks : int, optional
#             Number of blocks transferred so far [default: 1].
#         bsize  : int, optional
#             Size of each block (in tqdm units) [default: 1].
#         tsize  : int, optional
#             Total size (in tqdm units). If [default: None] remains unchanged.
#         """
#         if tsize is not None:
#             self.total = tsize  # pylint: disable=attribute-defined-outside-init
#         self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


# def download_url(url, filename):
#     """Download a file from url to filename, with a progress bar."""
#     with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
#         urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


class ConfigManager:
    def __init__(self, config: Dict):
        self.config = config

    def init_object(self, name: str, *args, **kwargs) -> object:
        # Root module
        root = "src"

        object_path = self.config[name]
        if object_path is None:
            return None

        module_name, object_name = object_path.rsplit(".", 1)

        module = import_module(f"{root}.{module_name}")

        object_args = self.config[f"{name}_args"] or {}
        kwargs = {**kwargs, **object_args}

        return getattr(module, object_name)(*args, **kwargs)

    def init_objects(self, name: str, *args, **kwargs) -> List[object]:
        # Root module
        root = "src"

        objects = []

        object_paths = self.config[name]
        n_objects = len(object_paths)
        object_args = self.config[f"{name}_args"] or [{}] * n_objects

        # Repeat single args across objects
        args = [arg if isinstance(arg, list) else [arg] * n_objects for arg in args]
        # print(args)
        args = list(zip(*args))
        # FIXME Figure out something for kwargs

        for object_path, object_arg, arg in zip(object_paths, object_args, args):
            module_name, object_name = object_path.rsplit(".", 1)
            module = import_module(f"{root}.{module_name}")

            objects.append(getattr(module, object_name)(*arg, **object_arg))

        return objects


def load_yaml(path):
    with open(path, "r") as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


# def _import_class(module_and_class_name: str) -> type:
#     """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
#     module_name, class_name = module_and_class_name.rsplit(".", 1)
#     module = importlib.import_module(module_name)
#     class_ = getattr(module, class_name)
#     return class_
