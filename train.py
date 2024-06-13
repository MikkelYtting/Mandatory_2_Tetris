import argparse
import os
import shutil
from random import random, randint, sample
from datetime import datetime
from multiprocessing import Process, Queue
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000) # højere decay = længere tids trænning
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epochs between testing phases")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to train concurrently")
    parser.add_argument("--max_blocks", type=int, default=50000, help="Maximum number of blocks to run")

    args = parser.parse_args()
    return args

def train_model(opt, model_id, queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(123 + model_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123 + model_id)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    state = state.to(device)

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    total_blocks = 0
    epsilon_values = []
    epoch_values = []
    while total_blocks < opt.max_blocks and epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        epsilon_values.append(epsilon)  # Log the epsilon value
        epoch_values.append(epoch)  # Log the epoch value
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=False)
        total_blocks += 1

        next_state = next_state.to(device)
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset().to(device)
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Model: {}, Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            model_id, epoch, opt.num_epochs, action, final_score, final_tetrominoes, final_cleared_lines))
        
        if epoch > 0 and epoch % opt.save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = "{}/tetris_model_{}_epoch_{}_{}".format(opt.saved_path, model_id, epoch, timestamp)
            torch.save(model, model_path)

    # Save the final model at the end of training
    model_path = "{}/tetris_model_{}_epoch_{}_final".format(opt.saved_path, model_id, epoch)
    torch.save(model, model_path)
    queue.put((model_id, final_score, model_path, epoch_values, epsilon_values))

def main(opt):
    processes = []
    queue = Queue()
    for model_id in range(opt.num_models):
        p = Process(target=train_model, args=(opt, model_id, queue))
        p.start()
        processes.append(p)

    best_model_id, best_score, best_model_path, best_epoch_values, best_epsilon_values = None, 0, None, [], []
    model_paths = []
    for _ in range(opt.num_models):
        model_id, score, model_path, epoch_values, epsilon_values = queue.get()
        model_paths.append(model_path)
        if score > best_score:
            best_model_id, best_score, best_model_path, best_epoch_values, best_epsilon_values = model_id, score, model_path, epoch_values, epsilon_values

    for p in processes:
        p.join()

    # Delete all models except the best one
    for path in model_paths:
        if path != best_model_path:
            os.remove(path)

    if best_model_id is not None:
        print(f"Best Model: {best_model_id} with score: {best_score}, saved at: {best_model_path}")
    else:
        print("No model reached the desired performance threshold.")

    # Plot the epsilon values for the best model
    plot_epsilon_values(best_epoch_values, best_epsilon_values)

def plot_epsilon_values(epoch_values, epsilon_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_values, epsilon_values, marker='o', linestyle='-', color='b')
    plt.title('Epsilon Value per Epoch for Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    opt = get_args()
    main(opt)
