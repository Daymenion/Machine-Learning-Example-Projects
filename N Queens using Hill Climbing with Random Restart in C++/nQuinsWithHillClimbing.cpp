//
// Created by Daymenion on 20.04.2022.
//
// N-Queen problem solving using Hill Climbing with Random Restarts by Mehmet ÜNLÜ
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License

#include <bits/stdc++.h> // For using standard C++ library
#include <random> // For using random number generator

using namespace std;

int n_queens; // number of queens
int restart_cnt = 0, max_restarts = 40; //restart_cnt is the number of times the algorithm has restarted and max_restarts is the maximum number of restarts
int cost = 0, instance = 0, maxInstance = 100, fail_cnt = 0; //cost is the cost of the current solution and instance is the number of the current instance
//cost is the cost of the current solution, instance is the number of the current instance and maxInstance is the maximum number of instances
bool flag = false; //flag is true if the current solution is the optimal solution
int moves = 0; //moves is the number of moves made in the current instance


//  fn to generate a random configuration of N queens on the chess board
void random_Board_Generator(
        vector<int> &chess_board) { //chess_board is the chess board state vector which is a vector of integers representing the chess board state
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n_queens - 1);
    for (int i = 0; i < n_queens; i++) {
        chess_board[i] = dis(gen);
    }
}

// fn to print soln: a 2-D Chess Board Configuration of the N-Queens
void print_ChessBoard(
        vector<int> &chess_board) { //chess_board is the chess board state vector which is a vector of integers representing the chess board state
    for (int i = 0; i < n_queens; i++) { //for each queen
        for (int j = 0; j < n_queens; j++) { //for each column
            if (chess_board[i] == j) { //if the queen is in the column
                cout << "Q "; //print a Q
            } else { //if the queen is not in the column
                cout << ". "; //print a dot
            }
        }
        cout << endl; //print a new line
    }
    cout << endl; //print a new line
}


// heuristic/objective func of a state: #pairs of queens attacking each other (diagonal, horizontal, vertical)
int heuristicFunc(
        vector<int> &chess_board) { //chess_board is the chess board state vector which is a vector of integers representing the chess board state
    int h = 0; //heuristic value
    for (int i = 0; i < n_queens; i++) {  //for each row (queen) in the chess board
        for (int j = i + 1; j < n_queens; j++) { //for each column (queen) in the chess board
            if (chess_board[i] == chess_board[j]) { //if the current queen and the other queen are in the same column
                h++; //increment the heuristic value
            }
            if (abs(chess_board[i] - chess_board[j]) ==
                abs(i - j)) { //if the current queen and the other queen are in the same diagonal
                h++; //increment the heuristic value
            }
        }
    }
    return h; //return the heuristic value
}


// fn to find the best neighbour state of the board with the lowest heuristic value out of all neighbours of the current state
vector<int> nextBoard_withRandomRestart(
        vector<int> &chess_board) { //chess_board is the chess board state vector which is a vector of integers representing the chess board state
    int bestCost, tempCost, currCost; // to store costs
    vector<int> next_board(n_queens), temp_board(n_queens); //to store the best neighbour state
    next_board = chess_board;  // initialise the next_state & Temp_state with Current_state of Chess_Board
    temp_board = chess_board;

    currCost = heuristicFunc(chess_board);  //heuristic value of current state of chess_board
    bestCost = currCost; // stores the best cost out of all neighbours. Initialised with current state cost
    bool changed = false; //flag to check if the best cost is updated or not in the loop

    //loop to find the best neighbour state of the current state of the chess board with the lowest heuristic value out of all neighbours
    for (int i = 0; i < n_queens; i++) { //for each queen in the i-th column
        for (int j = 0; j < n_queens; j++) { //for each queen in the j-th column
            if (i != j) { //if the queen is not in the same column as the current queen
                temp_board[i] = j; //swap the queen in the i-th column with the queen in the j-th column
                tempCost = heuristicFunc(temp_board); //heuristic value of the temp_board
                if (tempCost < bestCost) { //if the heuristic value of the temp_board is less than the bestCost
                    bestCost = tempCost; //update the bestCost
                    next_board = temp_board; //update the next_board
                    moves++; //increment the number of moves
                    changed = true;  //update the flag to false
                }
            }
        }
        temp_board[i] = chess_board[i]; //reset the temp_board to the current state of the chess_board
    }

    // IF: cost(current state) == cost(next-best state)
    // function didn't find any better neighbour state with better heuristic value than the current state
    // THEN: randomly generate a new configuration of Queens and take it to be the next-best board
    if (!changed) { //if the flag is true (i.e. the best cost is not updated)
        // then randomly generate a new configuration of Queens and take it to be the next-best board
        restart_cnt++; //random restart counter incremented


        if (restart_cnt >= max_restarts) { //if the random restart counter is greater than the restart limit
            // restart limit reached so finalize the instance (that's why flag=true) and return the current state
            flag = true;
            return next_board;
        }
        random_Board_Generator(next_board); // Generating a new Random Configuration of queens
        cost = heuristicFunc(next_board);     // and calculating its heuristic value
        moves += n_queens; //increment the number of moves by the number of queens because the new board is a new random state
    }

    // ELSE: cost(current state) != cost(next-best state) => update the cost and return the next-best state
    else
        cost = bestCost; //update the cost

    return next_board; //return the next_board
}


int main() { //main function

    // for using shuffle
    random_device rd;
    mt19937 g(rd());

    cout << "\tN-QUEEN PROBLEM\nHill Climbing with Random Restart" << endl;
    cout << "For Each Solution solve with maximum 100 instances and each instance maximum 40 random restart\n\n"<< endl;
    // INPUT
    cout << "Enter Number of queens:" << endl;
    cin >> n_queens; // Global Variable to store the Number of Queens for N-Queen Problem

    // n_queens have to grather than 4, you can solve the problem with 100 queens too but it will take too long to solve
    if (n_queens <= 3 && n_queens != 1) {
        cout << "No arrangement is possible" << endl;
        return 0;
    }

    float average_moves = 0.0;
    float average_restartCnt = 0.0;
    float average_execTime = 0.0;

    vector<int> chess_board(n_queens);// stores the Configuration of N-Queens on Chess_Board
    vector<int> goal_state[15]; // stores the Configuration of N-Queens on Goal_State
    float goal_states_stats[16][3] = {0.0}; // stores the Configuration of N-Queens on Final_Goal_State
    int cost_currState; // stores heuristic value of current state of the chess board
    int local_restart_cnt = 0; // local restart counter
    for (int i = 0; i < 15; i++) {
        clock_t start = clock(); // For execution time of program
        cost = 0, instance = 0, fail_cnt = 0;
        moves = 0; // number of moves
        local_restart_cnt = 0; // local restart counter
        while (instance <
               maxInstance) { // loop to generate many instances of n-queen and solve them ; #instances=maxInstance
            instance++;
            local_restart_cnt += restart_cnt; // local restart counter incremented
            restart_cnt = 0;
            flag = false; //initialising variables for each instance

            //generate a random board configuration of queens and shuffle it
            random_Board_Generator(chess_board); // Generating a new Random Configuration of queens
            shuffle(chess_board.begin(), chess_board.end(), g); //shuffling the board configuration

            cost_currState = heuristicFunc(
                    chess_board); // stores the heuristic value of the current state of chess_board

            // loop to find the next-best state and update the current state and its cost until the goal state is reached
            while (cost_currState != 0) {
                // cost != 0 => not the soln board yet
                // get the next better board and its heuristic cost and update the current state and its cost
                chess_board = nextBoard_withRandomRestart(chess_board);
                if (!flag) { // => a better neighbouring state found => update the current state and its cost and continue the loop to find the next-best state
                    cost_currState = cost;
                } else {// flag==true => no better neighbouring state found => restart the loop to find the next-best state
                    // Random Restart Limit reached for the current instance => restart the loop to find the next-best state
                    // terminate the current instance ; soln can't be found for this instance
                    fail_cnt++; //keeps track of the #instances which could not find a soln
                    flag = false; //initialising variables for each instance for the next instance
                    break; //restart the loop to find the next-best state for the next instance
                }
            }

            // check if Goal state reached
            if (cost_currState == 0) {
                // then => store the soln state of the chess board for display and stats
                clock_t stop = clock(); //stop the clock for execution time of program
                goal_state[i] = chess_board; //store the soln state of the chess board
                goal_states_stats[i][0] = (float) moves; // stores the number of moves to reach the goal state
                goal_states_stats[i][1] = (float) local_restart_cnt; //stores the #restart_cnt
                goal_states_stats[i][2] =
                        (float) (stop - start) / CLOCKS_PER_SEC; //stores the execution time of the program
            }
        }
        if (fail_cnt == maxInstance) // No soln Found
            cout << "\nCouldn't find a Solution\n";
    }

    // display the stats of the goal states and stores average stats
    for (int i = 0; i < 15; i++) {
        print_ChessBoard(goal_state[i]); //print the chess board configuration
        cout << "Number of moves to reach the goal state of " << i + 1 << "th solution: " << goal_states_stats[i][0]
             << endl;
        cout << "Number of Random Restarts: " << goal_states_stats[i][1] << endl;
        cout << "Execution Time: " << goal_states_stats[i][2] << endl;
        average_moves += goal_states_stats[i][0]; //stores the average number of moves to reach the goal state of all instances
        average_restartCnt += goal_states_stats[i][1];
        average_execTime += goal_states_stats[i][2];
    }

    goal_states_stats[15][0] =
            average_moves / (float) 15.0; //stores the average number of moves to reach the goal state
    goal_states_stats[15][1] = average_restartCnt / (float) 15.0; //stores the average #restart_cnt
    goal_states_stats[15][2] = average_execTime / (float) 15.0; //stores the average execution time of the program

    // display the stats of the goal states of the instances of n-queen problem for n=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

    cout << "\n\nFor Each Solution have to less than 100 instances and each instance has to less than 40 random restart"<< endl;
    cout << "Average Number of Moves, Restarts and Exec Time to reach the each goal states: " << endl;
    cout << "\t\tMove \t  Restart Count \tExec Time" << endl;
    for (int i = 0; i < 15; i++)
        cout << i + 1 << "th solution:\t" << goal_states_stats[i][0] << "\t\t" << goal_states_stats[i][1] << "\t\t"
             << goal_states_stats[i][2] << endl;
    cout << "Average: \t" << goal_states_stats[15][0] << "\t\t" << goal_states_stats[15][1] << "\t\t"
         << goal_states_stats[15][2] << endl;

    return 0;
}