Goal: Create Neural Network(s) to make a chess AI
Possible Network Structure:
Input: 8*8*12 bitmap implementation
	Rook * 2
	Knight * 2
	Pawn * 2
	King  * 2
	Queen * 2 
	Bishop * 2

	Output: [-1,1], 1 is White win, -1 is Black Win, 0 is draw
Target is result from game
	This could be good but has flaws
Other potential input: n-dimensional feature space from FEN, attempt various feed-foward structures
	Output: Same as before or a different evaluation function

train model on database of games
use value network to influence agent when purning search tree of moves
Monte Carlo Tree Search (MCTS)
where to start:
generate data.py
-load pgn files and create easy read in for State class
model.py
-create a value net with feature input and value output, this is our f(x)
state.py
-make a board that can be converted to a feature input, x, for our f(x)
game.py
-save this for the end: create a minmax or negamax agent, potentially use alpha-beta pruning along with our f(x) to find next moves
	-bonus: make this work with GUI

GOOD FEATURES:
-piece position
-player turn
-turn number
-number of possible moves
-see Chess Machine Learning.pdf for more features
Evaulation Function: https://en.wikipedia.org/wiki/Evaluation_function
Use evaluation as influence for features, not for target... maybe
if the value net doesn't converge, or more importantly learn, to the target when the target is the discrete win/loss/draw, we have to change target to a "more smooth" function

