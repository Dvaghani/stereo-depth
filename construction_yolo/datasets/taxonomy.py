"""
Unified 11-class taxonomy every dataset converter maps into.
Must stay identical to construction_dataset/dataset.yaml's class order.

Classes 8-10 (suspended-load, barrier, cable) were appended after the
original 8-class run so existing label files (ids 0-7) stay valid.
"""

CLASS_NAMES = ["container", "pole", "scaffolding", "crane",
               "person", "machinery", "vehicle", "building",
               "suspended-load", "barrier", "cable"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)
