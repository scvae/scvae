# ======================================================================== #
# 
# Copyright (c) 2017 - 2018 scVAE authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# ======================================================================== #

import os
import re
import sys
import shutil
import time
from math import floor

import urllib.request

# Time

def formatTime(t):
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))

def formatDuration(seconds):
    if seconds < 0.001:
        return "<1 ms"
    elif seconds < 1:
        return "{:.0f} ms".format(1000 * seconds)
    elif seconds < 60:
        return "{:.3g} s".format(seconds)
    elif seconds < 60 * 60:
        minutes = floor(seconds / 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        return "{:.0f}m {:.0f}s".format(minutes, seconds)
    else:
        hours = floor(seconds / 60 / 60)
        minutes = floor((seconds / 60) % 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

# Strings

def normaliseString(s):
    
    s = s.lower()
    
    replacements = {
        "_": [" ", "-", "/"],
        "": ["(", ")", ",", "$"]
    }
    
    for replacement, characters in replacements.items():
        pattern = "[" + re.escape("".join(characters)) + "]"
        s = re.sub(pattern, replacement, s)
    
    return s

def properString(original_string, translation, normalise = True):
    
    if normalise:
        transformed_string = normaliseString(original_string)
    else:
        transformed_string = original_string
    
    for proper_string, related_strings in translation.items():
        if transformed_string in related_strings:
            return proper_string
    
    return original_string

def capitaliseString(original_string):
    string_parts = re.split(
        pattern=r"(\s)",
        string=original_string,
        maxsplit=1
    )
    if len(string_parts) == 3:
        first_word, split_character, rest_of_original_string = string_parts
        if re.match(pattern=r"[A-Z]", string=first_word):
            capitalised_first_word = first_word
        else:
            capitalised_first_word = first_word.capitalize()
        capitalised_string = capitalised_first_word + split_character \
            + rest_of_original_string
    else:
        if re.match(pattern=r"[A-Z]", string=original_string):
            capitalised_string = original_string
        else:
            capitalised_string = original_string.capitalize()
    return capitalised_string

def enumerateListOfStrings(list_of_strings):
    if len(list_of_strings) == 1:
        enumerated_string = list_of_strings[0]
    elif len(list_of_strings) == 2:
        enumerated_string = " and ".join(list_of_strings)
    elif len(list_of_strings) >= 3:
        enumerated_string = "{}, and {}".format(
            ", ".join(list_of_strings[:-1]),
            list_of_strings[-1]
        )
    return enumerated_string

# IO

def copyFile(URL, path):
    shutil.copyfile(URL, path)

def removeEmptyDirectories(source_directory):
    for directory_path, _, _ in os.walk(source_directory, topdown = False):
        if directory_path == source_directory:
            break
        try:
            os.rmdir(directory_path)
        except OSError as os_error:
            pass
            # print(os_error)

def downloadFile(URL, path):
    urllib.request.urlretrieve(URL, path, download_report_hook)

def download_report_hook(block_num, block_size, total_size):
    bytes_read = block_num * block_size
    if total_size > 0:
        percent = bytes_read / total_size * 100
        sys.stderr.write("\r{:3.0f}%.".format(percent))
        if bytes_read >= total_size:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("\r{:d} bytes.".format(bytes_read))

# Shell output

RESETFORMAT = "\033[0m"
BOLD = "\033[1m"

def bold(string):
    """Convert to bold type."""
    return BOLD + string + RESETFORMAT

def underline(string, character="-"):
    """Convert string to header marks"""
    return character * len(string)

def heading(string, underline_symbol = "-", plain = False):
    string = "{}\n{}\n".format(string, underline(string, underline_symbol))
    if not plain:
        string = bold(string)
    return string

def title(string, plain = False):
    underline_symbol = "═"
    return heading(string, underline_symbol, plain)

def subtitle(string, plain = False):
    underline_symbol = "─"
    return heading(string, underline_symbol, plain)

def subheading(string, plain = False):
    underline_symbol = "╌"
    return heading(string, underline_symbol, plain)
