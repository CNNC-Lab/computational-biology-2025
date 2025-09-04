import os, sys
import urllib
from typing import Any, Union
import pickle
import re
import numpy as np
import h5py

from ..operations import isiterable
from .info import logger

data_path = None  # root path of the project's data folder
data_label = None  # global label of the experiment (root folder's name within the data path)
instance_label = None  # usually derived from the data_label, can contain extra parameters
filename_prefixes = None
paths = {}


def set_storage_locations(data_path_, data_label_, instance_label_=None, save=True):
    """
    Define paths to store data
    :param save: [bool] is False, no paths are created
    :return save_paths: dictionary containing all relevant storage locations
    """
    if save:
        logger.info("Setting storage paths...")
        main_folder = os.path.join(data_path_, data_label_)

        figures = main_folder + '/figures/'
        inputs = main_folder + '/inputs/'
        parameters = main_folder + '/parameters/'
        results = main_folder + '/results/'
        activity = main_folder + '/activity/'
        network = main_folder + '/system/'
        logs = main_folder + '/logs/'
        other = main_folder + '/other/'

        filename_prefixes_ = {
            'state_matrix': 'SM',
            'output_mapper': 'OM',
            'sequencer': 'SEQ',
            'embedding': 'EMB',
            'connectivity': 'CON'
        }

        dirs = {'main': main_folder, 'figures': figures, 'inputs': inputs, 'parameters': parameters,
                'results': results, 'activity': activity, 'logs': logs, 'other': other, 'system': network}

        for d in list(dirs.values()):
            try:
                os.makedirs(d)
            except OSError:
                pass

        dirs['label'] = data_label_

        global data_path
        global data_label
        global instance_label
        global filename_prefixes
        global paths

        data_path = data_path_
        data_label = data_label_
        if instance_label_ is not None:
            data_label = instance_label_
        instance_label = instance_label_
        filename_prefixes = filename_prefixes_
        paths = dirs

        return dirs
    else:
        logger.info("No data will be saved!")
        return {'label': False, 'figures': False, 'activity': False}


def remove_files(fname):
    """
    Remove all files in list
    :param fname:
    :return:
    """
    if isiterable(fname):
        for ff in fname:
            if os.path.isfile(ff) and os.path.getsize(ff) > 0:
                os.remove(ff)
    else:
        if os.path.isfile(fname) and os.path.getsize(fname) > 0:
            os.remove(fname)


def import_mod_file(full_path_to_module):
    """
    import a module from a path
    :param full_path_to_module:
    :return: imports module
    """
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    sys.path.append(module_dir)
    try:
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        return module_name, module_obj
    except Exception as er:
        raise ImportError("Unable to load module {0}, check if the name is repeated with other scripts in "
                          "path. Error is {1}".format(str(module_name), str(er)))


def save_pkl_object(obj: Any, filename: str) -> None:
    """Store objects as pickle files.

    Args:
        obj (Any): Object to pickle.
        filename (str): File path to store object in.
    """
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filename: str, **kwargs) -> Any:
    """Reload pickle objects from path.

    Args:
        filename (str): File path to load object from.

    Returns:
        Any: Reloaded object.
    """
    with open(filename, "rb") as input:
        obj = pickle.load(input, **kwargs)
    return obj


def write_to_hdf5(
        log_fname: str, log_path: str, data_to_log: Any, dtype: str = "S5000"
) -> None:
    """Writes data to an hdf5 file and specified log path within.

    Args:
        log_fname (str): Path of hdf5 file.
        log_path (str): Path within hdf5 file to store data at.
        data_to_log (Any): Data (array, list, etc.) to store at `log_path`
        dtype (str, optional): Data type to store as. Defaults to "S5000".
    """
    # Store figure paths if anywhere created
    if dtype == "S5000":
        try:
            data_to_store = [t.encode("ascii", "ignore") for t in data_to_log]
        except AttributeError:
            data_to_store = data_to_log
    else:
        data_to_store = np.array(data_to_log)

    h5f = h5py.File(log_fname, "a")
    if h5f.get(log_path):
        del h5f[log_path]
    h5f.create_dataset(
        name=log_path,
        data=data_to_store,
        compression="gzip",
        compression_opts=4,
        dtype=dtype,
    )
    h5f.flush()
    h5f.close()


def download_dataset(file_urls, target_paths):
    """
    Download a dataset from the provided urls
    :param file_urls: [list of str] complete urls
    :param target_paths: [list of str] target storage locations
    :return:
    """
    logger.info("Downloading the dataset... (It may take some time)")

    for url, pth in zip(file_urls, target_paths):
        if not os.path.isfile(pth):
            logger.info("\t - Downloading {0}".format(url))
            urllib.request.urlretrieve(url, pth)

    logger.info("Done!")


class FileIO(object):
    """
    Standard file loading and saving. Handle hickle (h5py + pickle) or pickle dictionaries, depending what's
    available. This class simplifies data handling by providing a simple wrapper for pickle (hickle) save and load
    routines
    """

    def __init__(self, filename, compression=None):
        """
        Create the file object
        :param filename: full path to target file
        :param compression:
        """
        self.filename = filename
        self.compression = compression

    def __str__(self):
        return "%s" % self.filename

    def load(self):
        """
        Loads h5-file and extracts the dictionary within it.

        Outputs:
          dict - dictionary, one or several pairs of string and any type of variable,
                 e.g dict = {'name1': var1,'name2': var2}
        """
        print("Loading %s" % self.filename)
        with open(self.filename, 'r') as f:
            data = pickle.load(f)
            return data

    def save(self, data):
        """
        Stores a dictionary (dict), in a file (filename).
        Inputs:
          filename - a string, name of file to store the dictionary
          dict     - a dictionary, one or several pairs of string and any type of variable,
                     e.g. dict = {'name1': var1,'name2': var2}
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)
