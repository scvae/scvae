MAJOR_VERSION = "2"
MINOR_VERSION = "0"
PATCH_VERSION = "0"
VERSION_SUFFIX = ""

VERSION = (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

__version__ = ".".join(map(str, VERSION))

if VERSION_SUFFIX:
    __version__ = "{}-{}".format(__version__, VERSION_SUFFIX)
