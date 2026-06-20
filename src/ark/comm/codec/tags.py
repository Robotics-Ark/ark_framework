from enum import IntEnum


VERSION = 1


class SpaceTag(IntEnum):
    BOX = 1
    DISCRETE = 2
    MULTI_BINARY = 3
    MULTI_DISCRETE = 4
    TEXT = 5
    DICT = 6
    TUPLE = 7
    SEQUENCE = 8
    GRAPH = 9
    ONE_OF = 10
    IMAGE = 11
    GRAYSCALE_IMAGE = 12
    RGB_IMAGE = 13
    DEPTH_IMAGE = 14
