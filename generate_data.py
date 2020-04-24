#!/usr/bin/env python
import os 
import numpy as np
import chess.pgn
import chess.svg
import io
from state import State
def generate_dataset():
    #create array of boards (X) with corresponding value (Y)
    X,Y =[],[]
    games = []
    game_number = 0
    target_values ={'1/2-1/2':0, '0-1':-1, '1-0':1}
    for fn in os.listdir("data"):
        try:
            pgn = open(os.path.join("data", fn),encoding="utf-8", errors="surrogateescape")
        except Exception:
            continue
        #read in each game in the file and put each move into the array, with the correct label
        #board array is a fen string representation of every position
        #we need to record values from pgn file because fen strings don't include it   
        for game in pgn:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                continue
            if game is None:
                break
            result = game.headers['Result']
            if result not in target_values:
                continue
            
            board = game.board()
            value = target_values[result] 
            for i, move in enumerate(game.mainline_moves()):
               
                if board.is_valid() is True:
                    board.push(move)
                    fen_string = board.fen()
                    Y.append(value)
                    #instead of apending fen_string, we need to transform each board 
                    #state to our representation for the neural network
                    #TODO:State Class
                    transform = State(board).board_transform()
                    X.append(transform)
                    #X.append(fen_string)
                else:
                    continue
            print("parsing game %d, got %d examples" % (game_number, len(X)))
            #enumerators
            games.append(game)
            game_number += 1
            if len(X) > 1000000:
                X = np.array(X)
                Y = np.array(Y)
                return X, Y
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
#encode features by vector representation


if __name__ == "__main__":
    datas_X, datas_Y = generate_dataset()
    np.savez_compressed("data/trainig_data_1M.npz", a=datas_X, b=datas_Y)
