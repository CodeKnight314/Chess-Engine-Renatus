import torch
import torch.nn as nn
import torch.optim as optim
import chess
import configs
import numpy as np
from collections import namedtuple
import argparse
from model import RenatusV2
from tqdm import tqdm
import random
from mct_search import MonteSearch, Node, get_state

class ReplayMemory:
    def __init__(self, capacity, device="cuda"):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition on the specified device."""
        transition = self.Transition(*args)
        transition = self.Transition(
            state=transition.state.to(self.device) if transition.state is not None else None,
            action=transition.action.to(self.device),
            next_state=transition.next_state.to(self.device) if transition.next_state is not None else None,
            reward=transition.reward.to(self.device)
        )
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def choose_legal_move(model: nn.Module, board: chess.Board, state: torch.Tensor, epsilon: float):
    if random.random() < epsilon:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)
    else:
        # Use MCTS to choose the best move
        root = Node(board)
        best_board = MonteSearch(root, model, num_iterations=100)
        return best_board.move_stack[-1]

def get_material_score(board: chess.Board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_score = sum(piece_values.get(piece, 0) * len(board.pieces(piece, chess.WHITE)) for piece in chess.PIECE_TYPES)
    black_score = sum(piece_values.get(piece, 0) * len(board.pieces(piece, chess.BLACK)) for piece in chess.PIECE_TYPES)
    return white_score - black_score if board.turn == chess.WHITE else black_score - white_score

def get_heuristic_score(board: chess.Board):
    """
    Combines multiple heuristics to calculate a score for the given board state.
    """
    # Material Score
    material_score = get_material_score(board)

    # Mobility Score (difference in number of legal moves)
    white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.push(chess.Move.null())
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()
    mobility_score = white_mobility - black_mobility if board.turn == chess.WHITE else black_mobility - white_mobility

    # King Safety (penalty for exposed king)
    king_safety_score = 0
    for color in chess.COLORS:
        king_square = board.king(color)
        if king_square is not None:
            attacks = board.attackers(not color, king_square)
            if attacks:
                king_safety_score -= len(attacks) * (1 if color == chess.WHITE else -1)

    w_material, w_mobility, w_king_safety = 1.0, 0.1, 0.5
    total_score = (w_material * material_score) + (w_mobility * mobility_score) + (w_king_safety * king_safety_score)

    return total_score

def get_reward(board: chess.Board, is_terminal: bool):
    if is_terminal:
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        else:
            return 0.0
    else:
        return get_heuristic_score(board)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RenatusV2(input_channels=27, num_blocks=19).to(device)
target_net = RenatusV2(input_channels=27, num_blocks=19).to(device)
target_net.load_state_dict(model.state_dict())
target_net.eval()

optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
memory = ReplayMemory(configs.MEMORY_SIZE)

def optimize_model():
    if len(memory) < configs.BATCH_SIZE:
        return

    transitions = memory.sample(configs.BATCH_SIZE)
    batch = configs.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)
    optimizer.zero_grad()
    
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Get policy and value from model
    policy, state_values = model(state_batch)
    # Ensure action indices are within bounds
    action_batch = torch.clamp(action_batch, 0, state_values.size(1) - 1)
    
    # Gather the state-action values based on the action indices
    state_action_values = state_values.gather(1, action_batch)
    
    # Compute next state values using target network
    target_net.eval()
    next_state_values = torch.zeros(configs.BATCH_SIZE, device=device)
    if non_final_next_states.size(0) > 0:
        _, next_state_values_model = target_net(non_final_next_states)
        next_state_values[non_final_mask] = next_state_values_model.max(1)[0].detach()

    expected_state_action_values = (next_state_values * configs.GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    loss.backward()
    optimizer.step()

def train(args):
    """
    Main training loop.
    """
    steps_done = 0

    if args.path:
        pretrained_dict = torch.load(args.path, weights_only=True)
        model_dict = model.state_dict()

        # Filter out unmatched keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        target_net.load_state_dict(model_dict, strict=False)

    for episode in tqdm(range(configs.NUM_EPISODES)):
        board = chess.Board()
        state = get_state(board)
        done = False

        while not done:
            # Select and perform an action
            epsilon = configs.EPSILON_END + (configs.EPSILON_START - configs.EPSILON_END) * np.exp(-1. * steps_done / configs.EPSILON_DECAY)
            action = choose_legal_move(model, board, state, epsilon)

            # Convert action (chess.Move) to a 1D index for the network output
            action_index = torch.tensor([[action.from_square * 64 + action.to_square]], 
                                       device=device, dtype=torch.long)

            # Make the move
            next_board = board.copy()
            next_board.push(action)
            next_state = get_state(next_board) if not next_board.is_game_over() else None
            reward = torch.tensor([get_reward(next_board, next_board.is_game_over())], device=device)
            done = next_board.is_game_over()

            # Store the transition in memory
            memory.push(state, action_index, next_state, reward)

            # Move to the next state
            state = next_state
            board = next_board
            steps_done += 1

            # Perform one step of the optimization (on the target network)
            optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if episode % configs.TARGET_UPDATE == 0:
            target_net.load_state_dict(model.state_dict())
            torch.save(target_net.state_dict(), f'target_net_episode_{episode}.pth')

        # Print episode information (optional)
        if episode % 100 == 0:
            print(f"Episode {episode}/{configs.NUM_EPISODES}, Steps: {steps_done}, Epsilon: {epsilon:.4f}")

    print('Complete')
    torch.save(model.state_dict(), 'renatus_chess_model.pth')  # Save the trained model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="Pretrained model weights if available")
    args = parser.parse_args()
    train(args)