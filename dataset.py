import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import chess
import chess.pgn
import numpy as np
from mct_search import get_state

# Standard piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # Not typically used in material evaluation
}

def move_to_policy(move):
    move_tensor = torch.zeros((4096))
    move_index = move.from_square * 64 + move.to_square
    move_tensor[move_index] = 1
    return move_tensor

def get_heuristic_value_target(board):
    """
    Heuristic function to evaluate the board state without an engine.
    Returns a value between -1 and 1 based on material balance.
    """
    white_score = sum(PIECE_VALUES[piece.piece_type] for piece in board.pieces(chess.PAWN, chess.WHITE))
    black_score = sum(PIECE_VALUES[piece.piece_type] for piece in board.pieces(chess.PAWN, chess.BLACK))

    for piece_type, value in PIECE_VALUES.items():
        white_score += len(board.pieces(piece_type, chess.WHITE)) * value
        black_score += len(board.pieces(piece_type, chess.BLACK)) * value

    # Normalize the score to [-1, 1]
    centipawn_value = (white_score - black_score) / 4000.0  # 4000 as an approximate maximum difference
    return max(-1.0, min(1.0, centipawn_value))

class PGNDataset(Dataset):
    def __init__(self, pgn_file: str):
        super().__init__()
        self.pgn_file = pgn_file
        self.positions = []
        
        with open(pgn_file) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                white_elo = game.headers.get("WhiteElo")
                black_elo = game.headers.get("BlackElo")
                if white_elo and black_elo:
                    avg_elo = (int(white_elo) + int(black_elo)) / 2
                    if avg_elo >= 2100:
                        board = chess.Board()
                        for move in game.mainline_moves():
                            current_state = get_state(board)
                            policy = move_to_policy(move)
                            board.push(move)
                            value = get_heuristic_value_target(board)
                            self.positions.append((current_state, policy, value))
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        current_state, policy, value = self.positions[index]
        return current_state, policy, torch.tensor(value, dtype=torch.float)

def custom_collate_fn(batch):
    current_states, policies, values = zip(*batch)
    current_states_batch = torch.stack(current_states)
    policies_batch = torch.stack(policies)
    values_batch = torch.stack(values)
    return current_states_batch, policies_batch, values_batch

def get_dataloader(file_path: str, batch_size: int):
    return DataLoader(PGNDataset(file_path), 
                      batch_size=batch_size, 
                      collate_fn=custom_collate_fn, 
                      shuffle=True, 
                      num_workers=os.cpu_count()//2)