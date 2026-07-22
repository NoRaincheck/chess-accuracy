import chess

from chess_accuracy.common import Division


def heuristic_division(num_plies):
    opening_end = 20
    endgame_start = 60

    middle = min(opening_end, num_plies) if num_plies > opening_end else None
    end = min(endgame_start, num_plies) if num_plies > endgame_start else None

    return Division(middle, end, num_plies)


def faithful_division(boards):
    indexed = list(enumerate(boards))

    def majors_and_minors(board):
        count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type not in (
                chess.KING,
                chess.PAWN,
            ):
                count += 1
        return count

    def backrank_sparse(board):
        white_back = 0
        for sq in chess.SQUARES:
            if chess.square_rank(sq) == 0:
                piece = board.piece_at(sq)
                if piece and piece.color == chess.WHITE:
                    white_back += 1
        black_back = 0
        for sq in chess.SQUARES:
            if chess.square_rank(sq) == 7:
                piece = board.piece_at(sq)
                if piece and piece.color == chess.BLACK:
                    black_back += 1
        return white_back < 4 or black_back < 4

    def mixedness_score(y, white, black):
        key = (white, black)
        if key == (0, 0):
            return 0
        if key == (1, 0):
            return 1 + (8 - y)
        if key == (2, 0):
            return 2 + (y - 2) if y > 2 else 0
        if key == (3, 0):
            return 3 + (y - 1) if y > 1 else 0
        if key == (4, 0):
            return 3 + (y - 1) if y > 1 else 0
        if key == (0, 1):
            return 1 + y
        if key == (1, 1):
            return 5 + abs(4 - y)
        if key == (2, 1):
            return 4 + (y - 1)
        if key == (3, 1):
            return 5 + (y - 1)
        if key == (0, 2):
            return 2 + (6 - y) if y < 6 else 0
        if key == (1, 2):
            return 4 + (7 - y)
        if key == (2, 2):
            return 7
        if key == (0, 3):
            return 3 + (7 - y) if y < 7 else 0
        if key == (1, 3):
            return 5 + (7 - y)
        if key == (0, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0

    def mixedness(board):
        total = 0
        for y in range(7):
            for x in range(7):
                white_count = 0
                black_count = 0
                for dy in range(2):
                    for dx in range(3):
                        rank = y + dy
                        file = x + dx
                        if rank < 8 and file < 8:
                            sq = chess.square(file, rank)
                            piece = board.piece_at(sq)
                            if piece:
                                if piece.color == chess.WHITE:
                                    white_count += 1
                                else:
                                    black_count += 1
                total += mixedness_score(y + 1, white_count, black_count)
        return total

    mid_game = None
    for idx, board in indexed:
        if (
            majors_and_minors(board) <= 10
            or backrank_sparse(board)
            or mixedness(board) > 150
        ):
            mid_game = idx
            break

    end_game = None
    if mid_game is not None:
        for idx, board in indexed:
            if majors_and_minors(board) <= 6:
                end_game = idx
                break

    if mid_game is not None and end_game is not None and mid_game >= end_game:
        mid_game = None

    return Division(
        middle=mid_game,
        end=end_game,
        plies=len(boards) - 1,
    )
