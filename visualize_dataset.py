import logging
import time
import collections
import os
import pathlib
import re
import string
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text as text

path = "dataset/train.tsv"
columns = ["PT", "AN"]

dataset = tf.data.experimental.CsvDataset(
    path,
    [tf.string, tf.string],
    field_delim="\t",
    header=True
)

for line in dataset:
    this_line = dict(zip(columns, line))

    print(f"\nPortuguese: {this_line['PT']}\nAngrarosskesh: {this_line['AN']}\n")