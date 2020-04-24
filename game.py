#!/usr/bin/env python
from __future__ import print_function
import os
import chess
import time
import chess.svg
import traceback
import base64
from state import State
import torch 
from model import Net
from progressbar import ProgressBar 
pbar = ProgressBar()
MAXVAL = 10000
class Valuator(object):
	def __init__(self):
		# import torch 
		# from model import Net
		device = torch.device('cuda')
		values = torch.load("value.pth", map_location = lambda storage, loc: storage)
		self.model = Net()
		self.model.load_state_dict(values)
		self.model.eval()



	def __call__(self, s):
		self.count += 1
		real_board = s.board_transform()[None]
		output = self.model(torch.tensor(real_board).float().cuda())
		return float(output.data[0][0])
	def reset(self):
		self.count = 0

def minimax(s, v, depth, a, b, big=False):		
	#white is max, black is min
	if depth >= 5 or s.board.is_game_over():
		return v(s)
	turn = s.board.turn
	if turn == chess.WHITE:
		ret = - MAXVAL
	else:
		ret = MAXVAL
	if big:
		bret = []

	#beam search
	sort_array = []
	for e in s.board.legal_moves:
		s.board.push(e)
		sort_array.append((v(s), e))
		s.board.pop()
	move = sorted(sort_array, key=lambda x: x[0], reverse=s.board.turn)

	if depth >= 3:
		move = move[:10]
	for e in [x[1] for x in move]:
		s.board.push(e)
		next_val = minimax(s, v, depth+1, a, b)
		s.board.pop()
		if big:
			bret.append((next_val, e))
		if turn == chess.WHITE:
			ret = max(ret, next_val)
			a = max(a, ret)
			if a >= b:
				break
		else:
			ret = min(ret, next_val)
			b = min(b, ret)
			if a >= b:
				break
	if big:
		return ret, bret
	else:
		return ret
def leaves(s, v):
	ret = []
	start = time.time()
	v.reset()
	val_1 = v(s)
	print("before the minmax")
	val_2, ret = minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
	eta = time.time() - start
	print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (val_1, val_2, v.count, eta, int(v.count/eta)))
	return ret

s = State()
v = Valuator()

def to_svg(s):
	return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')
from flask import Flask, Response, request
app = Flask(__name__)

@app.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())

def computer_move(s, v):
  # computer move
  print("beginning of move")
  move = sorted(leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  if len(move) == 0:
    return
  print("top 3:")
  for i,m in enumerate(move[0:3]):
    print("  ",m)
  print(s.board.turn, "moving", move[0][1])
  s.board.push(move[0][1])

# move given in algebraic notation
@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="")
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
  else:
    print("GAME IS OVER")
    response = app.response_class(
      response="game over",
      status=200
    )
    return response
  print("hello ran")
  return hello()
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response

if __name__ == "__main__":
	app.run(debug=True)
