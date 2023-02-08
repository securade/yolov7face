# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IPython compatibility layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import IPython


def get_ipython():
    """Returns IPython's InteractiveShell Instance.
    """
    return IPython.get_ipython()


def get_kernel():
    """Returns IPython kernel.
    """
    return get_ipython().kernel


def get_kernelapp():
    return get_ipython().kernel.parent


def in_ipython():
    """Returns True if we're in an IPython like environment.
    """
    ip = IPython.get_ipython()
    return hasattr(ip, 'kernel')
