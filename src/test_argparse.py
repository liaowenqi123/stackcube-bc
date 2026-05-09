#!/usr/bin/env python
import sys
print("DEBUG: Script started", flush=True)

import argparse
print("DEBUG: argparse imported", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, default=0)
args = parser.parse_args()

print("DEBUG: Parsed args:", args, flush=True)
print("Script finished successfully", flush=True)
