import argparse
import torch
import os
from src.tetris import Tetris
import matplotlib.pyplot as plt

def get_args():
    # Henter argumenter fra kommandolinjen
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    
    parser.add_argument("--width", type=int, default=10, help="Bredden på spillet")
    parser.add_argument("--height", type=int, default=20, help="Højden på spillet")
    parser.add_argument("--block_size", type=int, default=30, help="Størrelsen på en blok")
    parser.add_argument("--fps", type=int, default=300, help="Frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models", help="Stien til gemte modeller")
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
    plot_blocks_per_epoch(blocks_per_epoch, 'Trained Model Blocks per Epoch', 'g')

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

    # Opretter mappen hvis den ikke findes
    save_folder = 'plot_blocks_per_epoch'
    os.makedirs(save_folder, exist_ok=True)

    # Gemmer plottet
    plt.savefig(os.path.join(save_folder, 'blocks_per_epoch.png'))
    
    # Viser plottet
    plt.show()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
