"""Trainer-process PYTHONPATH dir. Not imported by the GUI.

``sitecustomize.py`` in this folder is loaded by the SimpleTuner subprocess
because ``train_runs._trainer_env`` puts this directory on PYTHONPATH.
"""
