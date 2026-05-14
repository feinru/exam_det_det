
import argparse
from dataclasses import dataclass

@dataclass
class Config:
    bidirectional: bool = True

parser = argparse.ArgumentParser()
parser.add_argument("--bidirectional", action="store_true")
args = parser.parse_args([]) # simulate no arguments

cfg = Config(bidirectional=args.bidirectional)
print(f"Config bidirectional: {cfg.bidirectional}")
