"""
Main entry point for DINOv2 training.
Simplified script that delegates to training module.
"""

import argparse
import torch.multiprocessing as mp
from pathlib import Path

from configs import get_args_parser
from training import train_dinov2


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        'Semantic-DINOv2 with Sequence Packing', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start training
    train_dinov2(args)