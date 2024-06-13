import argparse
import torch
import shutil
import os
import json
from src.tetris import Tetris
from random import randint
import matplotlib.pyplot as plt

def get_args():
    # Henter argumenter fra kommandolinjen
    parser = argparse.ArgumentParser(
        """Implementation of a Dummy Model to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="Bredden på spillet")
    parser.add_argument("--height", type=int, default=20, help="Højden på spillet")
    parser.add_argument("--block_size", type=int, default=30, help="Størrelsen på en blok")
    parser.add_argument("--fps", type=int, default=300, help="Frames per second")
    parser.add_argument("--num_epochs", type=int, default=10, help="Antal epochs at køre")
    parser.add_argument("--max_blocks", type=int, default=50000, help="Maksimalt antal blokke der kan bruges")
    parser.add_argument("--output_file", type=str, default="dummy_play_output/blocks_per_epoch.json", help="Fil til at gemme blokke per epoch")

    args = parser.parse_args()
    return args

def clear_log_dir(log_dir):
    # Rydder logmappen, hvis den eksisterer
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

def dummy_play(opt):
    # Sikrer at output-mappen eksisterer
    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)

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
            _, done = env.step(action, render=False)  # Slukker for rendering for hastighed
            num_blocks += 1
            total_blocks += 1

            if done:
                blocks_per_epoch.append(num_blocks)
                print(f"Epoch: {epoch}, Number of Blocks: {num_blocks}")
                break

    # Udskriver total antal blokke og gennemsnitlige blokke per epoch
    total_epochs = len(blocks_per_epoch)
    average_blocks_per_epoch = total_blocks / total_epochs if total_epochs > 0 else 0
    print(f"Total number of blocks used: {total_blocks}")
    print(f"Average number of blocks per epoch: {average_blocks_per_epoch:.2f}")

    # Plotter data
    plot_blocks_per_epoch(blocks_per_epoch, 'Dummy Model Blocks per Epoch', 'b')

    # Gemmer output i en fil
    save_blocks_per_epoch(blocks_per_epoch, opt.output_file)

    return blocks_per_epoch

def plot_blocks_per_epoch(blocks_per_epoch, title, color):
    # Plotter blokke per epoch
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

def save_blocks_per_epoch(blocks_per_epoch, output_file):
    # Sikrer at mappen eksisterer
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Gemmer blokke per epoch i en JSON-fil
    with open(output_file, 'w') as f:
        json.dump(blocks_per_epoch, f)
    print(f"Blocks per epoch saved to {output_file}")

if __name__ == "__main__":
    opt = get_args()
    blocks_per_epoch = dummy_play(opt)  # Fanger output i en array
    print("Blocks per epoch:", blocks_per_epoch)
