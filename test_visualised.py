import argparse
import torch
import os
from src.tetris import Tetris
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris with Visualization""")
    
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_blocks", type=int, default=50000, help="Maximum number of blocks to run")

    args = parser.parse_args()
    return args

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load(f"{opt.saved_path}/tetris_model_0_epoch_3000_final")
    else:
        model = torch.load(f"{opt.saved_path}/tetris_model_0_epoch_3000_final", map_location=lambda storage, loc: storage)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

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
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, done = env.step(action, render=True)  # Enable rendering to show the game
            num_blocks += 1
            total_blocks += 1

            if done:
                blocks_per_epoch.append(num_blocks)
                print(f"Epoch: {epoch}, Number of Blocks: {num_blocks}")
                break

    # Print total blocks and average blocks per epoch
    total_epochs = len(blocks_per_epoch)
    average_blocks_per_epoch = total_blocks / total_epochs if total_epochs > 0 else 0
    print(f"Total number of blocks used: {total_blocks}")
    print(f"Average number of blocks per epoch: {average_blocks_per_epoch:.2f}")

    # Plot the data
    plot_blocks_per_epoch(blocks_per_epoch, 'Trained Model Blocks per Epoch', 'g')

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

if __name__ == "__main__":
    opt = get_args()
    test(opt)
