import logging
import sys
import os

current_dir = os.getcwd()


def main() -> None:
    """
    The goal of this function is to have the logs being
    written at the root when the algorithm is launched.
    The logs are saved under the path KMeans/logs/logs.log

    Arguments:
        None
    Returns:
        None
    """

    if not hasattr(main, "called"):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        log_path = os.path.join(current_dir, "logs.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)
        logger.propagate = False
        setattr(main, "called", True)