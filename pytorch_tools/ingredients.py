""""""
import os
import copy
import tempfile
import random
import numpy as np

import sacred
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
import pymongo


FILE_DIR = os.path.dirname(__file__)
MONGODB_PORT = 27020
try:
    client = pymongo.MongoClient("localhost",
                                 serverSelectionTimeoutMS=1,
                                 port=MONGODB_PORT)
    client.server_info()
except pymongo.errors.ServerSelectionTimeoutError:
    MONGODB_PORT = None

#
# Torch Ingredient
#
torch_ingredient = sacred.Ingredient('torch_cfg')  # pylint: disable=invalid-name
torch_ingredient.add_config(os.path.join(FILE_DIR, 'config/torch.yaml'))


def cuda_is_available():
    # hack to check if cuda is available. calling torch.cuda.is_available in
    # this process breaks the multiprocesscing of multiple environments
    # See: https://github.com/pytorch/pytorch/pull/2811
    from torch.multiprocessing import Process, Queue
    def wrap_cuda_is_available(q):
        q.put(torch.cuda.is_available())
    q = Queue()
    p = Process(target=wrap_cuda_is_available, args=(q,))
    p.start()
    p.join()
    return q.get()


@torch_ingredient.config
def config(cuda, deterministic, benchmark):
    torch.backends.cudnn.fastest = not deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    if cuda is None:
        # cuda = torch.cuda.is_available()
        cuda = cuda_is_available()
    if cuda:
        # assert torch.cuda.is_available(), (
        #    "CUDA is not available. Please set "
        #    f"{torch_ingredient.path}.cuda=False.")

        assert cuda_is_available(), (
            "CUDA is not available. Please set "
            f"{torch_ingredient.path}.cuda=False.")


@torch_ingredient.capture
def set_random_seeds(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    # if torch.cuda.is_available():
    if cuda_is_available():
        torch.cuda.manual_seed_all(_seed)


@torch_ingredient.capture
def save_model_to_path(model, file_name, model_path, cuda):
    """Save PyTorch model to Observer (MongoDB)."""
    # model is not jsonpickleble. therefore it is saved as a file and
    # stored as a binary in the experiment database.
    if cuda:
        if isinstance(model, list):
            save_model = [copy.deepcopy(m).cpu()
                          if isinstance(m, nn.Module)
                          else m
                          for m in model]
        else:
            save_model = copy.deepcopy(model).cpu()

    torch.save(save_model, os.path.join(model_path, file_name))


@torch_ingredient.capture
def save_model_to_db(model, file_name, ex, cuda):
    """Save PyTorch model to Observer (MongoDB)."""
    # model is not jsonpickleble. therefore it is saved as a file and
    # stored as a binary in the experiment database.
    if cuda:
        if isinstance(model, list):
            save_model = [copy.deepcopy(m).cpu()
                          if isinstance(m, nn.Module)
                          else m
                          for m in model]
        else:
            save_model = copy.deepcopy(model).cpu()

    file_descriptor, tmp_file_path = tempfile.mkstemp()
    torch.save(save_model, tmp_file_path)

    ex.add_artifact(tmp_file_path, name=file_name)
    os.close(file_descriptor)


@torch_ingredient.capture
def load_model_from_db(ex, run_id, model_name, _log):
    run = ex.observers[0].runs.find_one({'_id': run_id})

    if run is None:
        _log.error(f"Loading model from non-existing run with id {run_id} is not possible.")
        exit()

    if model_name is None:
        model_name = [a['name']
                      for a in run['artifacts']
                      if a['name'].startswith("update_")][-1]
    elif not model_name.startswith("update_"):
        model_name = run['info'][model_name + "_name"]

    db_file_id = [m['file_id'] for m in run['artifacts']
                  if m['name'] == model_name][0]

    file_descriptor, tmp_file_path = tempfile.mkstemp()
    with open(tmp_file_path, 'a+b') as tmp_file:
        tmp_file.write(ex.observers[0].fs.get(db_file_id).read())
        tmp_file.seek(0, 0)
        model = torch.load(tmp_file, map_location=lambda s, l: s)
    os.close(file_descriptor)

    return model, model_name


@torch_ingredient.command(unobserved=True)
def print_config(print_config, _run, _log):
    """
    Prints configuration parameters.
    :param _run: Sacred Run object
    :type _run: Sacred.Run
    :param _log: Sacred logging module
    :type _log: Logger
    """
    # print config only for levels lower than warning
    if print_config and _log.getEffectiveLevel() < 30:
        sacred.commands.print_config(_run)


@torch_ingredient.pre_run_hook()
@torch_ingredient.capture()
def setup_db_logging(log_to_db, _run):
    """Setup saving experiments in database."""
    assert (MONGODB_PORT is not None or not log_to_db), (
        f"MongoDB server is not available at port {MONGODB_PORT}. "
        "Please start daemon or set log_to_db=False.")

    # Removes database entry if log_to_db=False
    if _run.observers and not log_to_db:
        mongo_observer = _run.observers[0]
        mongo_observer.runs.remove({'_id': mongo_observer.run_entry['_id']})
        _run.observers = []

    print_config()  # pylint: disable=no-value-for-parameter


