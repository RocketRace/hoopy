"""
The `hoopy` CLI transforms and runs a module directly.
"""

import argparse
import hoopy

# Example transformer
from hoopy.transform import transform
import sys

input = "\n".join(sys.stdin.readlines())
print(transform(input))
