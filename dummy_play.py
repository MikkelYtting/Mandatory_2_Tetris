import argparse
import torch
import cv2
import shutil
import os
from src.tetris import Tetris
from random import randint
from tensorboardX import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of a Dummy Model to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--log_path", type=str, default="dummy_tensorboard")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to run")
    parser.add_argument("--max_blocks", type=int, default=10000, help="Maximum number of blocks to run")

    args = parser.parse_args()
    return args

def clear_log_dir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

def dummy_play(opt):
    clear_log_dir(opt.log_path)  # Clear the log directory before starting the test

    writer = SummaryWriter(opt.log_path)
    total_blocks = 0
    epoch = 0
    blocks_per_epoch = []

    while total_blocks < opt.max_blocks:
        epoch += 1
        env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
        env.reset()
        num_blocks = 0
        done = False

        while not done and total_blocks < opt.max_blocks:
            next_steps = env.get_next_states()
            next_actions = list(next_steps.keys())
            action = next_actions[randint(0, len(next_actions) - 1)]
            _, done = env.step(action, render=False)  # Disable rendering for speed
            num_blocks += 1
            total_blocks += 1

            if done:
                # Log the number of blocks to TensorBoard at the end of the epoch
                writer.add_scalar('Dummy/Number_of_Blocks', num_blocks, epoch)
                blocks_per_epoch.append(num_blocks)
                print(f"Epoch: {epoch}, Number of Blocks: {num_blocks}")
                break

    writer.close()

    # Print total blocks and average blocks per epoch
    total_epochs = len(blocks_per_epoch)
    average_blocks_per_epoch = total_blocks / total_epochs if total_epochs > 0 else 0
    print(f"Total number of blocks used: {total_blocks}")
    print(f"Average number of blocks per epoch: {average_blocks_per_epoch:.2f}")

    # Plot the data
    plot_blocks_per_epoch(blocks_per_epoch, 'Dummy Model Blocks per Epoch', 'b')

    # Automatically inspect and print the logs after running the dummy model
    inspect_logs(opt.log_path)

def plot_blocks_per_epoch(blocks_per_epoch, title, color):
    average_blocks = sum(blocks_per_epoch) / len(blocks_per_epoch)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(blocks_per_epoch)), blocks_per_epoch, color=color, label='Blocks per Epoch')
    plt.axhline(y=average_blocks, color=color, linestyle='--', label=f'Average Blocks: {average_blocks:.2f}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Blocks')
    plt.legend()
    plt.grid(True)
    plt.show()

def inspect_logs(log_dir):
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    for event_file in event_files:
        event_path = os.path.join(log_dir, event_file)
        print(f"Inspecting {event_path}")
        for summary in summary_iterator(event_path):
            for value in summary.summary.value:
                print(value.tag, value.simple_value)

if __name__ == "__main__":
    opt = get_args()
    dummy_play(opt)
