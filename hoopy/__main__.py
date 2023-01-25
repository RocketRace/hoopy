"""
The `hoopy` CLI transforms and runs a module directly.
"""

# Example transformer
from hoopy.transform import transform
import sys

input = "\n".join(sys.stdin.readlines())
exec(transform(input))
