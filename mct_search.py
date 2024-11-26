import chess 
import numpy as np
from model import RenatusV2
import torch
from collections import defaultdict

def get_state(board: chess.Board):
    state = torch.zeros((27, 8, 8), dtype=torch.float32, device="cuda")

    # Encode piece types (12 planes: 6 for white pieces, 6 for black pieces)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            pieces = board.pieces(piece_type, color)
            for square in pieces:
                index = 6 * color + piece_type - 1
                row = square // 8
                col = square % 8
                state[index, row, col] = 1

    # Side to move (1 plane)
    if board.turn == chess.WHITE:
        state[12, :, :] = 1

    # Castling rights (2 planes: one for white, one for black)
    if board.has_kingside_castling_rights(chess.WHITE):
        state[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        state[13, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        state[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        state[14, :, :] = 1

    # En passant square (1 plane)
    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        state[15, row, col] = 1

    # Half-move clock (1 plane)
    state[16, :, :] = board.halfmove_clock / 100.0

    # Full-move number (1 plane)
    state[17, :, :] = board.fullmove_number / 100.0

    # Attack maps (8 planes: 4 for white, 4 for black)
    for color in chess.COLORS:
        for piece_type, attack_index_offset in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK], range(4)):
            pieces = board.pieces(piece_type, color)
            for square in pieces:
                attacks = board.attacks(square)
                for attack_square in attacks:
                    row = attack_square // 8
                    col = attack_square % 8
                    index = 18 + (4 * color + attack_index_offset)
                    state[index, row, col] = 1

    return state

class Node:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)  # Track untried moves to avoid redundant expansion
        self.is_fully_expanded = False
    
    def fully_explored_child(self):
        return self.is_fully_expanded
    
    def expand_child(self):
        if self.untried_moves:
            move = self.untried_moves.pop()
            new_board = self.board.copy()
            new_board.push(move)
            new_child = Node(new_board, self)
            self.children.append(new_child)
            if not self.untried_moves:
                self.is_fully_expanded = True
            return new_child
        return None
    
    def find_best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)))

def MonteSearch(root: Node, model: RenatusV2, num_iterations: int):
    node_map = defaultdict(lambda: None)  # Reuse previously calculated nodes
    
    for _ in range(num_iterations):
        node = root
        visited_nodes = []
        
        # Selection and expansion phase
        while node.fully_explored_child() and node.children:
            node = node.find_best_child()
            visited_nodes.append(node)
        
        if not node.fully_explored_child():
            expanded_node = node.expand_child()
            if expanded_node:
                node = expanded_node
                visited_nodes.append(node)
        
        # Evaluation phase
        if node_map[node.board.fen()] is not None:
            value = node_map[node.board.fen()]
        else:
            state = get_state(node.board).unsqueeze(0).to("cuda")
            _, score = model(state)
            value = score.item()
            node_map[node.board.fen()] = value  # Cache the evaluation value
        
        # Backpropagation phase
        for visited_node in visited_nodes:
            visited_node.visits += 1
            visited_node.value += value if visited_node.board.turn == root.board.turn else -value

    return root.find_best_child(exploration_weight=0.0).board