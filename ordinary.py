"""I pledge on my honor that I have not given or received
any unauthorized assistance on this project.
Evan Deist
"""
from reversi import *
import math

class Node:
    """A tree node that contains the board state, a dictionary to all child nodes, and UCT statistics"""
    def __init__(self, board, parent):
        self.board = board # current board state
        self.parent = parent # parent node
        self.children = dict() # maps a (x,y) tuple to a node

        # when this node is first explored, statistics are initialized
        self.r = 0.00
        self.t = 0.00
        self.q = 0.00

# We need two sets of global variables, in case this program is run against itself
b1 = getNewBoard()
resetBoard(b1)
b2 = getNewBoard()
resetBoard(b2)

tree1 = Node(b1, None)
tree2 = Node(b2, None)

# Keep track of last move made so we know how to descend tree
playerLastMove = None

def reset():
    """Resets global variables (for automatic testing purposes)"""
    global tree1
    global tree2
    global b1
    global b2
    global playerLastMove
    b1 = getNewBoard()
    resetBoard(b1)
    b2 = getNewBoard()
    resetBoard(b2)

    tree1 = Node(b1, None)
    tree2 = Node(b2, None)

    playerLastMove = None

def get_move(board, tile):
    """Descends the game tree based on whatever move was made, then runs a monte carlo rollout, then returns the best move"""
    global tree1
    global tree2
    global b1
    global b2
    global playerLastMove

    tree = tree1 if tile == 'X' else tree2
    b = b1 if tile == 'X' else b2 
    opponentTile = 'O' if tile == 'X' else 'X'

    # First we check if the board state is different, so we can
    # tell whether this is a consecutive get_move() call or not.
    # If this is the beginning of our turn, we descend the tree
    playerMove = None
    opponentMove = None
    if(b != board):
        for x in range(8):
            for y in range(8):
                if (board[x][y] != b[x][y]):
                    playerMove = playerLastMove
                    # if a previously blank tile has the opponent's chip,
                    # we know what move they made
                    if b[x][y] == ' ' and board[x][y] == opponentTile:
                        opponentMove = (x,y)
                    b[x][y] = board[x][y]
        # descend tree based on previous moves
        if playerMove != None:
            if playerMove in tree.children.keys():
                tree = tree.children[playerMove]
            else: # unexplored, so create new tree
                tree = Node(board, None)

        if opponentMove != None:
            if opponentMove in tree.children.keys():
                tree = tree.children[opponentMove]
            else: # unexplored, so create new tree
                tree = Node(board, None)

        # update globals
        if tile == 'X':
            tree1 = tree
            b1 = b
        else:
            tree2 = tree
            b2 = b

    # Run UCT to update node statistics
    UCT(tree, True, tile, opponentTile)

    # Select child with highest utility
    bestReward = -1
    bestMove = None
    for move in tree.children.keys():
        if not isValidMove(board, tile, move[0], move[1]):
            break
        childNode = tree.children[move]
        if childNode.r > bestReward:
            bestReward = childNode.r
            bestMove = move
    playerLastMove = (bestMove[0], bestMove[1])
    
    return (bestMove[0], bestMove[1])


def UCT(currNode, maxNode, playerTile, opponentTile):
    """recursivly descends one path of the tree and updates each nodes' r and t values"""
    value = None
    tile = playerTile if maxNode else opponentTile
    
    # check if this is a terminal node, and if it is, get the utility
    if getValidMoves(currNode.board, tile) == []:
        scores = getScoreOfBoard(currNode.board)
        playerScore = scores[playerTile]
        opponentScore = scores[opponentTile]
        # evaluate u(x)
        if (playerScore > opponentScore): # win: u(x) = 1
            value = 1.00
        elif (playerScore < opponentScore): # loss: u(x) = 0
            value = 0.00
        else: # tie: u(x) = 0.5
            value = 0.50

    else: # nonterminal node, get value of a child with UCB_choose
        (currNode, nextMove) = UCB_choose(currNode, maxNode, playerTile, opponentTile)
        nextNode = currNode.children[nextMove]
        # if this is a max node, the next level down will be min, and vice versa
        value = UCT(nextNode, not maxNode, playerTile, opponentTile)

    # update statistics
    currNode.r = (currNode.r * currNode.t + value) / (currNode.t + 1.00)
    currNode.t += 1.00

    return value

def UCB_choose(currNode, maxNode, playerTile, opponentTile):
    """Selects an unexplored node to try, if any. Otherwise, selects a child with best q value"""

    # Find all unexplored nodes, if any
    tile = playerTile if maxNode else opponentTile
    possibleMoves = getValidMoves(currNode.board, tile)
    
    # Lists are not hashable. We turn the moves into tuples so they 
    # can be used as dict keys
    possibleMoves = [tuple(x) for x in possibleMoves]
    unexploredMoves = []

    for move in possibleMoves:
        # A move has been explored if it is a child of this node
        if move not in currNode.children.keys():
            unexploredMoves.append(move)

    # If there are unexplored moves, try a random one
    if unexploredMoves != []:
        tryMove = random.choice(unexploredMoves)
        newBoard = getBoardCopy(currNode.board)
        makeMove(newBoard, tile, tryMove[0], tryMove[1])
        currNode.children[tryMove] = Node(newBoard, currNode)
        return (currNode, tryMove)

    
    # If all moves have been explored, choose the best one
    argmax = None
    maxVal = -1
    for move in possibleMoves:
        childNode = currNode.children[move]
        # Calculate q value
        value = childNode.r if maxNode else (1.00-childNode.r)
        value += math.sqrt((2.00 * math.log(currNode.t)) / childNode.t)
        childNode.q = value
        if (value > maxVal):
            maxVal = value
            argmax = move

    return (currNode, argmax)