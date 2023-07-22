import yaml
import os


def load_conf(path: str, include=False) -> yaml:
    """
    The goal of this function is to load the
    params
    Arguments:
        -path: str The path of configuration file to be loaded
        -include: bool: True if the user wants to charge a meta
        file with all configs params
    Returns:
        -file: yaml file The configuration file loaded
    """
    if include:
        with open(path, "r") as f:
            file = yaml.load(f, Loader)

    else:
        with open(path, "r") as ymlfile:
            file = yaml.safe_load(ymlfile)
    return file


def clean_params(dictionnary: dict) -> dict:
    """
    The goal of this function is, once the params are
    loaded, to clean the dictionnary containing them with
    no sub-dictionnary

    Arguments:
        dictionnary: dict: The dictionnary containing the params

    Returns:
        clean_dict: dict: The dictionnary once cleaned
    """

    for key in list(dictionnary.keys()):
        try:
            for subkey in dictionnary[key].keys():
                dictionnary[subkey] = dictionnary[key][subkey]
        except AttributeError:
            pass

    for key in list(dictionnary.keys()):
        if type(dictionnary[key]) == dict:
            del dictionnary[key]

    clean_dict = dictionnary

    return clean_dict


class Loader(yaml.SafeLoader):
    """
    The goal of this class is to make it possible to add
    all configs file with yml extension in the same file"""

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, "r") as f:
            return yaml.load(f, Loader)


Loader.add_constructor("!include", Loader.include)