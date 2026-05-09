#!/usr/bin/env python
import sys
print("DEBUG: test script started", flush=True)

print("DEBUG: importing argparse...", flush=True)
import argparse
print("DEBUG: argparse OK", flush=True)

print("DEBUG: importing torch...", flush=True)
import torch
print("DEBUG: torch OK", flush=True)

print("DEBUG: importing models...", flush=True)
sys.path.insert(0, "src")
from models import BCDiffusion, BCDiffusionTemporal, BCDiffusionKoopman, EMA
print("DEBUG: models OK", flush=True)

print("DEBUG: importing train_bcdiffusion...", flush=True)
import train_bcdiffusion
print("DEBUG: train_bcdiffusion OK", flush=True)

print("DEBUG: calling main()...", flush=True)
train_bcdiffusion.main()
print("DEBUG: main() finished", flush=True)
