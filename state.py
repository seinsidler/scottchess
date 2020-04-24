#!/usr/bin/env python
import chess
import numpy as np

class State(object):
	def __init__(self, board=None):
		if board is None:
			self.board = chess.Board()
		else:
			self.board = board
	def board_transform(self):
		#returns n-dimensional vector which is our neural nets peception vector of the board state
		#X = np.concatenate(self.side_to_move(),castling_rights(),
		#piece_positions(),sliding_piece_mobility(),attack_map(),defend_map())
		#1+4+64+12+64+64 = 209 bits
		X = np.concatenate((self.side_to_move(), self.castling_rights(), self.piece_positions(), self.material(), self.attack_map(), self.defend_map()))
		return X
	#helper functions
	#side_to_move()
	#castling_rights()
	#piece_positions()
	#material()
	#sliding_piece_mobility()
	#attack_map()










	#defend_map()
	def side_to_move(self):
		#length 1
		vec = np.zeros(1, dtype='uint8')
		vec[0] = self.board.turn*1.0
		return vec

	def castling_rights(self):
		#length 4
		vec = np.zeros(4, dtype='uint8')
		if self.board.has_queenside_castling_rights(chess.WHITE):
			vec[0] = 1
		else:	
			vec[0] = 0
		if self.board.has_kingside_castling_rights(chess.WHITE):
			vec[1] = 1
		else:
			vec[1] = 0
		if self.board.has_queenside_castling_rights(chess.BLACK):
			vec[2] = 1
		else:
			vec[2] = 0
		if self.board.has_kingside_castling_rights(chess.BLACK):
			vec[3] = 1
		else:
			vec[3] = 0
		return vec

	def piece_positions(self):
		#length 64
		vec = np.zeros(64, dtype='uint8')
		pn = 0
		for i in range(64):
			pp = self.board.piece_at(i)
			
			# if pp is not None:
	  #       	#print(i, pp.symbol())
			# 	vec[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[pp.symbol()]
	  #       #else:
	  #       #	vec[i] = 0
	  #   	else:
	  #   		vec[i] = 0
			if pp is not None:
				vec[i] = {"P": 1, "N": 2, "B": 3,"R":4, "Q": 5, "K": 6, "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[pp.symbol()]
			else:
				vec[i] = 0

		return vec
	def material(self):
		#length 12
		vec = np.zeros(12, dtype='uint8')
		pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]		
		for i in range(64):
			pp = self.board.piece_at(i)
			for j, piece in enumerate(pieces):
				if pp== piece:
					vec[j] += 1
		return vec
	def attack_map(self):
		#length 64
		piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100, "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
		vec = np.zeros(64, dtype='uint8')
		turn = True
		if self.side_to_move() == 0:
			turn = True
		else:
			turn = False
		for i in range(64):
			attackers = self.board.attackers(turn, i)
			values = []
			for j in range(64):
				if j in attackers:
					pp = self.board.piece_at(j)
					values.append(piece_values[pp.symbol()])
			if len(values) == 0:
				vec[i] = 0
			else:
				vec[i] = max(values)
		return vec

	def defend_map(self):
		#length 64
		piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100, "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
		vec = np.zeros(64, dtype='uint8')
		turn = True
		if self.side_to_move() == 1:
			turn = True
		else:
			turn = False
		for i in range(64):
			attackers = self.board.attackers(turn, i)
			values = []
			for j in range(64):
				if j in attackers:
					pp = self.board.piece_at(j)
					values.append(piece_values[pp.symbol()])
			if len(values) == 0:
				vec[i] = 0
			else:
				vec[i] = max(values)
		return vec





#HIGHSCORE: parsing game 10123, got 830646 examples
#cap at 1.2 mil