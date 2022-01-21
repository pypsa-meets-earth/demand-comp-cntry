import copy
import sys, os, pickle, json, hashlib
from functools import wraps

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_OUT"))



def cached_with_io(func):

    func.cache_pickle_path = os.path.join(os.environ.get("PROJECT_CACHE"), func.__name__ + '.p')

    if os.path.exists(func.cache_pickle_path):
        func.cache = pickle.load(open(func.cache_pickle_path, "rb"))
    else:
        func.cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):

        args_temp = copy.deepcopy(args)
        kwargs_temp = copy.deepcopy(kwargs)

        # convert to list for ease of indexing
        args_temp = list(args_temp)

        try:
            # try to dump a json with both args and kwargs
            args_string = json.dumps([args_temp, kwargs_temp])

        except TypeError:
            # so far, we found typeerrors when some of the args are numpy arrays. We need to convert these to lists
            # we are not currently trying to convert kwargs, since those are usually simpler data types.
            n_args = len(args_temp)
            for i in range(n_args):
                if type(args_temp[i]).__module__ == 'numpy':
                    args_temp[i] = args_temp[i].tolist()

            args_temp = tuple(args_temp)
            args_string = json.dumps([args_temp, kwargs_temp])

        args_hash = hashlib.sha256(args_string.encode('utf-8')).hexdigest()

        try:
            return func.cache[args_hash]

        except KeyError:

            # run function and save cache
            func.cache[args_hash] = func(*args)

            # save updated cache to picklefile
            os.makedirs(os.environ.get("PROJECT_CACHE"), exist_ok=True)
            pickle.dump(func.cache, open(func.cache_pickle_path, "wb"), protocol=4)

            return func.cache[args_hash]

    return wrapper