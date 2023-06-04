#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>

using namespace std;

class Board {
private:
  // By default, whie goes first.
  int current_move_color = 1;
  // 0 is empty, -1 is black, 1 is white.
  // board_state is accesed board_state[distance from left edge][distance from bottom].
  // So in the standard 7 x 6 board, the top right would be accesed board_state[6][5].
  vector<vector<int>> board_state;
  int board_width, board_height;
public:
  Board (vector<int> board_input, int board_width, int board_height) :
    board_width(board_width), board_height(board_height),
    board_state(board_width, vector<int> (board_height)) {
    set_board_state (board_input);
  }
  
  void set_board_state (vector<int> board_input) {
    int current_column = 0;
    int current_height = 0;
    for (int spot = 0; spot < board_input.size(); spot++) {
      if (current_column >= board_width) {
	// If it is past the last column, then stop filling up the board.
	break;
      }
      if (current_height >= board_height) {
	// If the column is full, go to the next column.
	current_column++;
	current_height = 0;
      }
      if (board_input[spot] == 0) {
	// 0 means go to next column, so if there is a 0, fill in the rest of the column as 0 to mean blank.
	for (int temp_height = current_height; temp_height < board_height; temp_height++) {
	  board_state[current_column][temp_height] = 0;
	}
	current_column++;
	current_height = 0;
      } else {
	board_state[current_column][current_height] = board_input[spot];
	current_height++;
      }
    }
  }
  
  vector<int> get_board_state () const {
    vector<int> output;
    for (const auto &column : board_state) {
      for (const auto &square : column) {
	if (square != 0) {
	  output.push_back(square);
	} else {
	  output.push_back(0);
	  break;
	}
      }
    }
    
    return output;
  }

  bool in_board (int column, int row) const {
    if (column < 0 || column >= board_width) {
      return false;
    }
    if (row < 0 || row >= board_height) {
      return false;
    }
    
    return true;
  }
  
  bool is_possible_move (int move) const {
    if (!in_board(move, 0)) return false;
    if (board_state[move][board_height - 1] != 0) return false;
    return true;
  }
  
  int move (int move) {
    if (!is_possible_move(move)) return -1;
    // If it is a possible move, play the move.
    for (auto &square : board_state[move]) {
      // Loop over each square in the column.
      if (square == 0) {
	// If the square is empty, play the move, swap the color, then exit the loop.
	square = current_move_color;
	current_move_color *= -1;
	return 0;
      }
    }
    
    return -1;
  }

  int player () const {
    return current_move_color;
    /*    int player = 0;
    for (const auto &column : board_state) {
      for (const auto &square : column) {
	// For each square in the board, get its value and add it to player.
	// This will make it so that in the end, player will be equal to number of white pieces - number of black pieces.
	player += square;
      }
    }

    // It is white's move if the number of squares are equal, it is blacks move if white has one more square than black.
    return (player == 0 ? 1 : -1);*/
  }

  vector<int> actions () const {
    vector<int> actions;

    int column_index = 0;
    for (const auto &column : board_state) {
      if (column[board_height - 1] == 0) {
	// For each column, if the top square is empty, that means that it is a legal move.
	actions.push_back(column_index);
      }
      column_index++;
    }
    
    return actions;
  }

  bool stalemate () const {
    for (const auto &column : board_state) {
      for (const auto &square : column) {
	// Loop over each square in the board.
	if (square == 0) {
	  // If there is an empty square, it is not stalemate.
	  return false;
	}
      }
    }

    // If there are no empty squares, then it is not stalemate.
    return true;
  }
  
  int number_of_lines (int color, int length) const {
    // I will loop over each square, and for each square go in every direction length - 1 times.
    // If each square inluding the original is the same as color, then I add 1 to the counter.
    // I return the counter at the end.
    int counter = 0;
    
    for (int column = 0; column < board_width; column++) {
      for (int row = 0; row < board_width; row++) {
	// Loop over each square.
	if (board_state[column][row] != color) {
	  // If the square is the wrong color, then skip to the next square.
	  continue;
	}
	for (int x_direction = -1; x_direction < 2; x_direction++) {
	  for (int y_direction = -1; y_direction < 2; y_direction++) {
	    // Check each direction.
	    //cout << "column: " << column << " row: " << row << " x_direction: " << x_direction << " y_direction: " << y_direction << "\n";
	    if (x_direction == 0 && y_direction == 0) {
	      // If the direction is just the current square, then its a waste to check so I just go to the next direction.
	      continue;
	    }

	    int line_end_column = column + ((length - 1) * x_direction);
	    int line_end_row = row + ((length - 1) * y_direction);
	    if (!in_board(line_end_column, line_end_row)) {
	      // If the very end of the line in this direction is off the board, then skip to the next direction.
	      continue;
	    }

	    // Assume that the direction is valid until proven otherwise.
	    bool valid_direction = true;
	    
	    for (int distance = 1; distance < length; distance++) {
	      // Go in the given direction for length squares.
	      // Distance starts at 1 because I don't have to check the color of the original square.
	      int temp_column = column + (distance * x_direction);
	      int temp_row = row + (distance * y_direction);
	      
	      if (board_state[temp_column][temp_row] != color) {
		// If the color doesn't match the original square color, then skip to the next direction.
		valid_direction = false;
		break;
	      }
	    }
	    
	    if (valid_direction) {
	      // If the code got to this spot, it means it passed all the tests for there not being a line of set length.
	      // That means that there was a line at this spot, so add one to the counter.
	      counter++;
	    }
	  }
	}
      }
    }

    // I divide counter by 2 because when counting the lines, the function will count a line from both ends.
    return counter/2;
  }

  bool terminal () const {
    if (number_of_lines(-1, 4) >= 1 || number_of_lines(1, 4) >= 1) {
      // Return true if a player has won.
      return true;
    }
    
    // Return true if it is a stalemate, otherwise return false.
    return stalemate();
  }
  
  int value () const {
    if (stalemate()) return 0;
    if (number_of_lines(1, 4) >= 1) return 1000;
    if (number_of_lines(-1, 4) >= 1) return -1000;
    
    int value = 0;

    value += number_of_lines(1, 1);
    value += number_of_lines(1, 2);
    value += number_of_lines(1, 3);

    value -= number_of_lines(-1, 1);
    value -= number_of_lines(-1, 2);
    value -= number_of_lines(-1, 3);

    if (value < -1000 || value > 1000) {
      cout << "big nono\n";
    }
    
    return value;
  }
  
  void print_board () const {
    for (const auto &column : board_state) {
      cout << "_";
    }
    cout << "\n";
    for (int row = board_height - 1; row >= 0; row--) {
      for (int column = 0; column < board_width; column++) {
	if (board_state[column][row] == 0) {
	  cout << " ";
	} else if (board_state[column][row] == -1) {
	  cout << "B";
	} else if (board_state[column][row] == 1) {
	  cout << "W";
	}
	// cout << column << " " << row << "  ";
      }
      cout << "|\n";
    }
    for (const auto &column : board_state) {
      cout << "-";
    }
    cout << "\n";
  }
};

int next_board_state (const Board &current, Board &next, const int &action) {
  next = current;
  return next.move(action);
}

int minimax (const Board current, int depth) {
  // White (1) is the max player, black (-1) is the min player.
  const int max_player = 1;
  const int min_player = -1;
  int value;
  
  //  cout << depth << "\n";
  
  if (current.terminal() || depth <= 0) return current.value();
  
  if (current.player() == max_player) {
    value = -INT_MAX;
    for (const auto &action : current.actions()) {
      Board next({}, 7, 6);
      if (next_board_state(current, next, action) < 0) {
	return -INT_MAX;
      }
      value = max(value, minimax(next, depth - 1));
    }
    return value;
  }
  
  if (current.player() == min_player) {
    value = INT_MAX;
    for (const auto &action : current.actions()) {
      Board next({}, 7, 6);
      if (next_board_state(current, next, action) < 0) {
	return -INT_MAX;
      }
      value = min(value, minimax(next, depth - 1));
    }
    return value;
  }
  
  // If something breaks, return -INT_MAX.
  return -INT_MAX;
}

int best_next_move (const Board &current, int depth) {
  const int max_player = 1;
  const int min_player = -1;

  int best_move;
  
  if (current.player() == max_player) {
    int value = -INT_MAX;
    for (const auto &action : current.actions()) {
      Board next({}, 7, 6);
      if (next_board_state(current, next, action) < 0) {
	return -INT_MAX;
      }

      int this_value = minimax(next, depth - 1);
      int next_value = max(value, this_value);
      cout << action << " " << this_value << "\n";
      
      if (next_value > value) {
	value = next_value;
	best_move = action;
      }
    }
  }
  
  if (current.player() == min_player) {
    int value = INT_MAX;
    for (const auto &action : current.actions()) {
      Board next({}, 7, 6);
      if (next_board_state(current, next, action) < 0) {
	return -INT_MAX;
      }

      int this_value = minimax(next, depth - 1);
      int next_value = min(value, this_value);
      cout << action << " " << this_value << "\n";
      
      if (next_value < value) {
	value = next_value;
	best_move = action;
      }
    }
  }
  
  return best_move;
}

int main () {
  int depth = 7;
  vector<int> board_state_input = {};

  Board minimax_test(board_state_input, 7, 6);
  minimax_test.print_board();
      
  while (1) {
    cout << "Player to move is: " << minimax_test.player() << "\n";
    int move;
    
    if (minimax_test.player() == 1) {
      cin >> move;
    } else if (minimax_test.player() == -1) {
      move = best_next_move(minimax_test, depth);
    }
    
    minimax_test.move(move);
    minimax_test.print_board();
    
    if (minimax_test.terminal()) {
      cout << "Game over, the value is: " << minimax_test.value() << "\n";
      break;
    }
  }
  
  return 0;
}
