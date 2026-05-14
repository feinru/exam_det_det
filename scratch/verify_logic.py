
import argparse
from train import TrainConfig

def test_config(bidirectional_flag=None, no_bidirectional_flag=None):
    class Args:
        def __init__(self, b, nb):
            self.bidirectional = b
            self.no_bidirectional = nb
            self.feature_root = "f"
            self.crop_root = "c"
            self.output_dir = "o"
            self.seq_len = 60
            self.labeling = "any"
            self.hidden_dim = 128
            self.num_layers = 2
            self.fc_dim = 64
            self.dropout = 0.3
            self.no_attention = False
            self.epochs = 50
            self.batch_size = 32
            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.no_pos_weight = False
            self.weighted_sampler = False
            self.patience = 10
            self.min_delta = 1e-4
            self.seed = 42
            self.device = "auto"
            self.num_workers = 0

    args = Args(bidirectional_flag, no_bidirectional_flag)
    
    # Logic from train.py
    bidirectional_val = args.bidirectional if args.bidirectional is not None else (False if args.no_bidirectional else TrainConfig.bidirectional)
    
    cfg = TrainConfig(
        bidirectional=bidirectional_val
    )
    return cfg.bidirectional

print(f"Default (None, None)      -> {test_config(None, None)} (Expected: True)")
print(f"Flag --bidirectional      -> {test_config(True, None)} (Expected: True)")
print(f"Flag --no-bidirectional   -> {test_config(None, True)} (Expected: False)")
