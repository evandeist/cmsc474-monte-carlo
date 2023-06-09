## Project 2 supervisor code
This is a test code for project 2.
It has 4 files.
1. reversi.py - it contains code for Reversi in python that I have picked up from http://inventwithpython.com/chapter15.html .
2. supervisor.py - this is the Supervisor whose pseudo code has been provided in the project PDF.
3. computer.py - this is the Computer player, against which you can try playing.
4. random_player.py - this is a Random player, that plays random moves. You can play against this too.


Importantly, in your submission file, you need to have a function 
`get_move(board, tile)`
where
- board is a 2D list of size 8x8 containing the characters ' ','X', or 'O' which correspond to black spot, Player1 and Player2 respectively.
- tile - is either 'X' or 'O', signifying who is supposed to move.
So, the function get_move() takes the current state of the board as well as who moves next and returns the next move which is the list of 2 integers in the range of [0,7]. eg. you may return [1,7] but [9,3] is invalid because it is an invalid board location.
Recall that this function will be called several times per each round until the time limit is reached.
Note that you can maintain a global variable in your program to gradually find a better move.


Now to run the superviser.py, you need to execute - 

`python supervisor.py <player1> <player2> <timeout_threshold> <verbose>`
or
`python3 supervisor.py <player1> <player2> <timeout_threshold> <verbose>`
depending on your environment.
In here,
- player1 (required)- represents the program running player 1. eg. with the current file you can have player1=computer or player1=random.
- player2 (required)- represents the program running player 2. 
- timeout_threshold  (optional) - number of seconds to wait for one move. Default set to 1.
- verbose (optional) - whether to display the board after each move or not. Default set to 1.

To run your own player, copy that file to the same directory and use them in the argument list.
For example, if you have your code in ucb.py, then you may run the following command -

    `python supervisor.py computer ucb`

Notes - 

- While displaying the board has indexing [(1-8),(1-8)] and even the moves are displayed like that. But internally everything works in indexing starting from zero.
- Remember to have the get_move() signature correct in your file. That is the only thing that is going to get called.
Remember to remove the .py from your filename when you are passing that as a player to supervisor.py.
- You can also use the same program against itself. eg  `python supervisor.py computer computer` is valid.
If you find any bugs in the supervisor code, please contact me via suhoshin@umd.edu and I will update the code accordingly (possibly I can ask the instructor about giving some bonus points regarding the bug report)



