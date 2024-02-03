#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

class Board {
private:
  // Helps with making the accesing of the bitboards more readable.
  // white is the max player, black is the min player.
  const int white = 0;
  const int black = 1;
  
  // There are 2 bitboards, one is all of blacks moves, the other is all of whites moves.
  // bitboard[0] is white, bitboard[1] is black.
  unsigned long long bitboards[2] = {0, 0};
  
  // Stores the amount to shift the board by to look for lines.
  // Vertical, Horizontal, Diagonal \, Diagonal /.
  int directions[4] = {1, 7, 6, 8};
  
  // Stores the height of where a piece would fall according to its column.
  int heights[7] = {0, 7, 14, 21, 28, 35, 42};
  // Stores the number of moves that have been done so far.
  int move_count = 0;
  int board_width = 7, board_height = 6;
  // Stores all the moves that have been played.
  int moves[42];
public:
  bool valid_move (int move) const {
    // If move is outside of the board, it is not a valid move.
    if (move < 0 || move > 6) return false;
    
    unsigned long long top = 0b1000000100000010000001000000100000010000001000000ULL;
    
    if ((top & (1ULL << heights[move])) == 0) return true;
    
    return false;
  }
  
  vector<int> valid_moves () const {
    vector<int> moves;
    
    for (int column = 0; column < 7; column++) {
      if (valid_move(column)) moves.push_back(column);
    }
    
    return moves;
  }
  
  int do_move (int column) {
    // If it isn't a valid move, then exit the function.
    if (!valid_move(column)) return -1;
    // column gives the index of the position of the move in height for that column.
    // Shift a bit into that position and store it in move.
    // After, increment the height by 1.
    // move now contains a 1 just in the location of the move.
    unsigned long long move = 1ULL << heights[column]++;
    // If move_count is even, then its rightmost bit is a 0.
    // This means that move_count & 1 will be 1 if move_count is odd, or 0 if move_count is even.
    // If move_count is even, it is white's move, if move_count is odd it is black's move.
    // bitboard[move_count & 1] access the bitboard of the current move.
    // ^= move is a bitwise XOR.
    // bitboards only gets affected in the spot where move is 1, which is where the move was played.
    bitboards[move_count & 1] ^= move;
    // Add the move to the list of moves, and then increment the number of moves by 1.
    moves[move_count++] = column;
    
    return 0;
  }
  
  void undo_move () {
    // Decrement move_count (the number of moves done) by 1.
    // Then find the column of the move done by looking into moves.
    int column = moves[--move_count]; // Reverses the 3rd step of do_move.
    // Decrement the height of the column by one.
    // Shift a bit into the position of the move being undone in move.
    long move = 1ULL << --heights[column]; // Reverses the 1st step of do_move.
    // move_count & 1 gives whether the move was done by white or black.
    // ^= move makes the spot where the move was done on the bitboard get turned into a 0.
    bitboards[move_count & 1] ^= move; // Reverses the 2nd step of do_move.
  }
  
  bool stalemate () const {
    for (int column = 0; column < 6; column++) {
      // For each move, if it is a valid move, that means that the column is empty.
      // If there is an empty column, that means that it is not a stalemate.
      if (valid_move(column)) return false;
    }
    
    // If none of the columns are empty, then it is a stalemate.
    return true;
  }
  
  bool win (bool player_black) const {
    for (const auto &direction : directions) {
      // bitboards[0] is the white board, bitboards[1] is the black board.
      // If the player is black, player_black will be true, which means that bitboards[player_black] will be bitboards[1].
      // Shifting the bitboard in the specific direction is like getting a line of 4 in a row, and compressing it into one square.
      // If the compressed squares are combined with an AND, then a compressed square will only be 1 if there was a 4 in a row in that spot.
      if ((  (bitboards[player_black] >> direction * 0)
	   & (bitboards[player_black] >> direction * 1)
	   & (bitboards[player_black] >> direction * 2)
	   & (bitboards[player_black] >> direction * 3)) != 0) return true;
    }
    
    return false;
  }
  int number_of_lines (bool player_black, int length) const {
    int line_count = 0;
    
    for (const auto& direction : directions) {
      // Shift the bitboard in the specific direction to check for lines.
      unsigned long long shifted_board = bitboards[player_black];

	//unsigned long long temp_board;
      for (int distance = 1; distance < length; distance++) {
	// Distance starts at 1, because starting at a distance of 0 makes it an and against itself, which will just output itself.
	shifted_board &= (bitboards[player_black] >> (direction * distance));
	/*
	  temp_board = shifted_board;
	if (direction == 7) {
	  cout << "Line length: " << length << "\n";
	  cout << "Distance: " << distance << "\n";
	  cout << "Direction: " << direction << "\n";
	  cout << "Current line count: " << line_count << "\n";
	  cout << shifted_board << "\n";
	  cout << "a";
	  cout << "Previous board: ";
	  print_bitboard(temp_board);
	  cout << "Board being anded: ";
	  print_bitboard(bitboards[player_black] >> (direction * distance));
	  cout << "end";
	  print_bitboard(shifted_board);
	  }*/
      }
      //print_bitboard(shifted_board);
      
      // Each 1 in shifted_board corresponds to a line, so if I count all the 1s, then I get the number of lines.
      line_count += __builtin_popcount(shifted_board);
    }

    return line_count;
  }
  
  bool terminal () const {
    // The game is over if a player has won.
    if (win(white)) return true;
    if (win(black)) return true;
    // The game is over if it is a stalemate.
    if (stalemate()) return true;
    
    // Otherwise the game is not over.
    return false;
  }
  
  int value () const {
    // white is the max player, black is the min player.
    // If black has won, return a very negative number.
    if (win(black)) return -2000;
    // If white has won, return a very positive number.
    if (win(white)) return 2000;
    // If it is a stalemate, the position is equal.
    if (stalemate()) return 0;
    
    // Starting the value as 1 because the player who goes first has an advantage.
    int value = 1;
    
    // Since white is max, add the white players advantages.
    value += 2 * number_of_lines(white, 2);
    value += 3 * number_of_lines(white, 3);
    
    // Since black is min, subtract the black players advantages.
    value += 2 * number_of_lines(black, 2);
    value += 3 * number_of_lines(black, 3);
    
    return value;
  }
  
  bool player_is_min () {
    // bitboards[0] is white, bitboards[1] is black.
    // white is the max player, black is the min player.
    // The player to move is white if move_count is even.
    // move_count is even if the last bit is a 0.
    // If the last bit of move_count is 0, then move_count & 1 will return 0.
    // That means that if player is white -> move_count is even -> return 0.
    //                    player is black -> move_count is odd --> return 1.
    return (move_count & 1);
  }
  
  void print_board () const {
    for (int i = 0; i < 7; i++) {
      cout << "-";
    }
    cout << "\n";
    
    for (int row = 5; row >= 0; row--) {
      for (int column = 0; column < 7; column++) {
	if  	(bitboards[white] >> (column * 7 + row) & 1) cout << "W"; // If white's board has a 1, print a W.
	else if (bitboards[black] >> (column * 7 + row) & 1) cout << "B"; // If black's board has a 1, print a B.
	else                                                 cout << " "; // If both boards have a 0, print nothing.
      }
      cout << "|\n";
    }
    
    for (int i = 0; i < 7; i++) {
      cout << "-";
    }
    cout << "\n";
  }
  
  void print_bitboard (unsigned long long bitboard) const {
    for (int i = 0; i < 7; i++) {
      cout << "=";
    }
    cout << "\n";
    
    for (int row = 5; row >= 0; row--) {
      for (int column = 0; column < 7; column++) {
	if (bitboard >> (column * 7 + row) & 1) cout << "1";
	else cout << " ";
      }
      cout << "||\n";
    }
    
    for (int i = 0; i < 7; i++) {
      cout << "=";
    }
    cout << "\n";
  }
};

// This function is mostly for testing purposes.
void play_moves (Board board, int moves[], int num_moves) {
  for (int move = 0; move < num_moves; move++) {
    board.do_move(moves[move]);
    board.print_board();
    cout << "Player to play next: " << board.player_is_min() << "\n";
    cout << "Value: " << board.value() << "\n";
  }
}

int minimax (Board state, int depth, int alpha, int beta) {
  // If the game is over, or it is reached the maximum depth, return the value.
  if (state.terminal() || depth <= 0) return state.value();
  
  int value;
  if (!state.player_is_min()) {
    // If it is the max player.
    value = -INT_MAX;
    for (const auto &action : state.valid_moves()) {
      state.do_move(action);
      value = max(value, minimax(state, depth - 1, alpha, beta));
      
      alpha = max(value, alpha);
      if (beta <= alpha) break;
      
      state.undo_move();
    }
    return value;
  }
  
  if (state.player_is_min()) {
    // If it is the min player.
    value = INT_MAX;
    for (const auto &action : state.valid_moves()) {
      state.do_move(action);
      value = min(value, minimax(state, depth - 1, alpha, beta));
      
      beta = min(value, beta);
      if (beta <= alpha) break;
      
      state.undo_move();
    }
    return value;
  }

  // If something broke, just return -INT_MAX.
  cout << "something is broken\n";
  return -INT_MAX;
}

int best_next_move (Board &current_board, int depth) {
  bool current_is_min = current_board.player_is_min();
  int alpha = -INT_MAX;
  int beta = INT_MAX;

  // Store all the move values.
  int move_values[7];
  for (int move = 0; move < 7; move++) {
    if (current_board.valid_move(move)) {
      // For each move, if the move is valid, then store its value.
      current_board.do_move(move);
      move_values[move] = current_board.value();
      current_board.undo_move();
    } else {
      // Otherwise store a bad value.
      // INT_MAX is too big/small for the sorting algorithm, so move it closer to 0 by 1.
      if (!current_is_min) {
	// For the max player, -INT_MAX is the worst value.
	move_values[move] = -INT_MAX + 1;
      } else {
	// For the min player, INT_MAX is the worst value.
	move_values[move] = INT_MAX - 1;
      }
    }
  }

  // Sort all the move values.
  int move_order[7];
  for (int order_index = 0; order_index < 7; order_index++) {
    if (current_is_min) {
      // Sort the moves by lowest value to highest value.
      int best = INT_MAX;
      for (int value_index = 0; value_index < 7; value_index++) {
	if (move_values[value_index] < best) {
	  move_order[order_index] = value_index;
	  best = move_values[value_index];
	}
      }
      move_values[move_order[order_index]] = INT_MAX;
    } else {
      // Sort the moves by highest value to lowest value.
      int best = -INT_MAX;
      for (int value_index = 0; value_index < 7; value_index++) {
	if (move_values[value_index] > best) {
	  move_order[order_index] = value_index;
	  best = move_values[value_index];
	}
      }
      move_values[move_order[order_index]] = -INT_MAX;
    }
  }

  cout << "Move Order:\n";
  for (const auto &move : move_order) {
    cout << move << "\n";
  }
  
  int best_move;
  int this_value;
  int best_value;
  if (!current_is_min) {
    best_value = -INT_MAX;
    // Call minimax on all the moves.
    for (const auto &move : move_order) {
      // All the illegal moves are at the end.
      if (!current_board.valid_move(move)) break;

      current_board.do_move(move);
      this_value = minimax(current_board, depth - 1, alpha, beta);
      cout << move << " " << this_value << "\n";
      if (this_value > best_value) {
	// If the new move value is better than the old best move value:
	// This is the new best move value.
	best_value = this_value;
	// This is the new best move.
	best_move = move;
      }
      current_board.undo_move();
    }
  } else {
    best_value = INT_MAX;
    // Call minimax on all the moves.
    for (const auto &move : move_order) {
      // All the illegal moves are at the end.
      if (!current_board.valid_move(move)) break;

      current_board.do_move(move);
      this_value = minimax(current_board, depth - 1, alpha, beta);
      cout << move << " " << this_value << "\n";
      if (this_value < best_value) {
	// If the new move value is better than the old best move value:
	// This is the new best move value.
	best_value = this_value;
	// This is the new best move.
	best_move = move;
      }
      current_board.undo_move();
    }
  }

  cout << "Move chosen: " << best_move << "\n";
  return best_move;
}


int main () {
  Board my_board;
  my_board.print_board();
  
  int test[] = {0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0, 3, 0, 4, 1, 2, 6, 3, 6, 2, 6, 2, 6, 3, 5, 4, 4, 4, 5, 5, 5, 5};
  int test2[] = {3, 3, 3, 3, 0, 3, 0, 4, 5, 5, 5, 5, 5, 4, 6, 4, 6, 4, 6, 4, 6};
  
  //  play_moves(my_board, test2, sizeof(test2)/sizeof(test2[0]));
  
  while (1) {
    int input;
    if (my_board.player_is_min()) {
      input = best_next_move(my_board, 11);
    } else {
      cin >> input;
    }
    my_board.do_move(input);
    my_board.print_board();
    if (my_board.terminal()) {
      cout << "Game Over\n";
      break;
    }
  }
  
  return 0;
}
