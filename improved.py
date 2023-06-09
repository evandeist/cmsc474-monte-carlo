"""I pledge on my honor that I have not given or received
any unauthorized assistance on this project.
Evan Deist
"""
from reversi import *
import math

class Node:
    """A tree node that contains the board state, a dictionary to all child nodes, and UCT statistics"""
    def __init__(self, board):
        self.board = board # current board state
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

tree1 = Node(b1)
tree2 = Node(b2)

# Keep track of last move made so we know how to descend tree
playerLastMove = None

# Reset globals (for automated testing purposes)
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

    tree1 = Node(b1)
    tree2 = Node(b2)

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
            else: # unexplored somehow, so create new tree
                tree = Node(board)

        if opponentMove != None:
            if opponentMove in tree.children.keys():
                tree = tree.children[opponentMove]
            else: # unexplored somehow, so create new tree
                tree = Node(board)

        # update globals
        if tile == 'X':
            tree1 = tree
            b1 = b
        else:
            tree2 = tree
            b2 = b

    # run UCT to update statistics
    UCT(tree, True, tile, opponentTile)

    # Select child with highest utility
    bestReward = -9999
    bestMove = None
    for move in tree.children.keys():
        if not isValidMove(board, tile, move[0], move[1]):
            break
        
        # Here's where the improvements come in:

        childNode = tree.children[move]
        i = move[0]
        j = move[1]
        value = childNode.r

        # InnerScore 
        aWeight = 0.4
        innerScore = (aWeight * value) / math.sqrt(((move[0] - 3.5) ** 2) + ((move[1] - 3.5) ** 2))

        # Greed Penalty
        scores = getScoreOfBoard(childNode.board)
        xScore = scores['X']
        oScore = scores['O']
        bWeight = 1.5
        greedPenalty = ((1 if tile == 'X' else -1) * (xScore - oScore))/((xScore + oScore)**bWeight)

        # Position Penalty
        cWeight = 0.4
        dWeight = 0.4
        eWeight = -1 # negative weight here because good moves are incentivized, not penalized
        positionPenalty = 0.00
        # "square" positions
        if (i == 1 and j == 1):
            positionPenalty = cWeight * value
        if (i == 1 and j == 6):
            positionPenalty = cWeight * value
        if (i == 6 and j == 1):
            positionPenalty = cWeight * value
        if (i == 6 and j == 6):
            positionPenalty = cWeight * value
        # "c-square" positions
        if (i == 0 and j == 1):
            positionPenalty = dWeight * value
        if (i == 1 and j == 0):
            positionPenalty = dWeight * value
        if (i == 6 and j == 0):
            positionPenalty = dWeight * value
        if (i == 7 and j == 1):
            positionPenalty = dWeight * value
        if (i == 7 and j == 6):
            positionPenalty = dWeight * value
        if (i == 6 and j == 7):
            positionPenalty = dWeight * value
        if (i == 0 and j == 6):
            positionPenalty = dWeight * value
        if (i == 1 and j == 7):
            positionPenalty = dWeight * value
        # if corner available, take it!
        if (i == 0 and j == 0):
            positionPenalty = eWeight * value
        if (i == 7 and j == 0):
            positionPenalty = eWeight * value
        if (i == 0 and j == 7):
            positionPenalty = eWeight * value
        if (i == 7 and j == 7):
            positionPenalty = eWeight * value

        value += innerScore - greedPenalty - positionPenalty

        if value > bestReward:
            bestReward = value
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
        currNode.children[tryMove] = Node(newBoard)
        return (currNode, tryMove)

    
    # If all moves have been explored, choose the best one
    argmax = None
    maxVal = -99999
    for move in possibleMoves:
        childNode = currNode.children[move]
        value = childNode.r if maxNode else (1.00-childNode.r)
        value += math.sqrt((2.00 * math.log(currNode.t)) / childNode.t)
        childNode.q = value
        if (value > maxVal):
            maxVal = value
            argmax = move

    return (currNode, argmax)
