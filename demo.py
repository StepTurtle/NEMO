#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate one password with the model for its exact length."""

import os
import sys
import math

from configs.configure import Configure
from ngram.ngram_creator import NGramCreator


def load_model(password_length, config):
    if password_length not in config.LENGTHS:
        raise ValueError("No configured model for password length {}.".format(password_length))

    model = NGramCreator({
        "name": "Demo",
        "alphabet": config.ALPHABET,
        "ngram_size": config.NGRAM_SIZE,
        "training_file": "input/" + config.TRAINING_FILE,
        "encoding": config.ENCODING,
        "encoding_errors": config.ENCODING_ERRORS,
        "length": password_length,
        "progress_bar": False
    })

    training_basename = os.path.basename(config.TRAINING_FILE)
    model_path = os.path.join(
        "trained",
        os.path.splitext(training_basename)[0] + "_cp_list_{}_{}.pack".format(
            config.NGRAM_SIZE,
            password_length
        )
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing model file: {}".format(model_path))

    model.load("ip_list")
    model.load("cp_list")
    model.load("ep_list")
    return model


def evaluate_password(password, model):
    if not model._is_in_alphabet(password):
        raise ValueError("Password contains characters outside the configured alphabet.")

    ip = password[:model.ngram_size - 1]
    ep = password[-(model.ngram_size - 1):]
    probability = model.ip_list[model._n2iIP(ip)]
    probability *= model.ep_list[model._n2iIP(ep)]

    old_position = 0
    for new_position in range(model.ngram_size, len(password) + 1):
        ngram = password[old_position:new_position]
        probability *= model.cp_list[model._n2iCP(ngram)]
        old_position += 1

    return probability


def probability_to_entropy(probability):
    if probability <= 0.0:
        raise ValueError("Probability must be greater than zero.")
    return -math.log2(probability)


def main():
    password = "?p5r6j"
    if not password:
        print("Password must not be empty.", file=sys.stderr)
        return 1

    config = Configure({"name": "Demo"})
    if len(password) < config.NGRAM_SIZE:
        print(
            "Password length {} is shorter than the configured n-gram size {}.".format(
                len(password), config.NGRAM_SIZE
            ),
            file=sys.stderr
        )
        return 1

    if len(password) not in config.LENGTHS:
        print(
            "Password length {} is not in the configured model lengths {}. Use one of: {}".format(
                len(password), config.LENGTHS, config.LENGTHS
            ),
            file=sys.stderr
        )
        return 1

    try:
        model = load_model(len(password), config)
        probability = evaluate_password(password, model)
        entropy = probability_to_entropy(probability)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1

    print("Password length: {}".format(len(password)))
    print("Probability: {:.16e}".format(probability))
    print("Self-information: {:.4f} bits".format(entropy))

if __name__ == "__main__":
    sys.exit(main())
