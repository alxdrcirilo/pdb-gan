from concurrent.futures import ThreadPoolExecutor
from random import choice
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import urllib.request

try:
    os.remove("logger.log")
except FileNotFoundError:
    pass

import logging
log = logging.getLogger("rich")
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])

from pdb_gan.fetch import Fetcher
from pdb_gan.preprocess import Preprocessor