import argparse
import torch
import os
from src.tetris import Tetris
import matplotlib.pyplot as plt
import cv2

def get_args():
    # Henter argumenter fra kommandolinjen
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris with Visualization""")
    
    parser.add_argument("--width", type=int, default=10, help="Bredden på spillet")
    parser.add_argument("--height", type=int, default=20, help="Højden på spillet")
    parser.add_argument("--block_size", type=int, default=30, help="Størrelsen på en blok")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models", help="Sti til gemte modeller")
    parser.add_argument("--max_blocks", type=int, default=50000, help="Maksimalt antal blokke der kan bruges")

    args = parser.parse_args()
    return args

def test(opt):
    # Initialiserer tilfældig frø
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Læsser modellen
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

    # Ensure the demo directory exists
    demo_path = 'demo'
    os.makedirs(demo_path, exist_ok=True)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(demo_path, 'tetris_gameplay.avi'), fourcc, opt.fps, (opt.width * opt.block_size * 2, opt.height * opt.block_size))

    # Kører spillet indtil max_blocks er nået
    while total_blocks < opt.max_blocks:
        epoch += 1
        env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
        env.reset()
        num_blocks = 0
        done = False

        while not done and total_blocks < opt.max_blocks:
            # Får næste tilstande og handlinger
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, done = env.step(action, render=True, video=out)  # Pass the video writer

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
    plot_blocks_per_epoch(blocks_per_epoch, 'Trained Model Blocks per Epoch', 'g')

    # Release the video writer
    out.release()
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    opt = get_args()
    test(opt)
