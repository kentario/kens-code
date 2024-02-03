#include <iostream>

using namespace std;


const int white = -1;
const int black = 1;
int board[8][8];

void setup_board ()
{
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      board[i][j] = 0;
    }
  }
  board[3][3] = white;
  board[4][4] = white;
  board[4][3] = black;
  board[3][4] = black;
}

bool inside_board (int x, int y)
{
  if (x < 0 || x > 7 || y < 0 || y > y) {
    return false;
  }
  return true;
}

bool is_legal_move (int x, int y, int color)
{
  // Not a legal move if the square is outside of the board, or is not empty.
  if (!inside_board(x, y) || !board[x][y] == 0) {
    return false;
  }
  // Check in all eight directions.
  for (int rel_y = -1; rel_y < 2; rel_y++) {
    for (int rel_x = -1; rel_x < 2; rel_x++) {
      int temp_y = y + rel_y;
      int temp_x = x + rel_x;
      // Don't check where the piece is being placed.
      if (rel_y == 0 && rel_x == 0) {
	continue;
      }
      int found = 0;
      // Continue going in a direction where the peices are opposite colors until something different is reached.
      while (inside_board(temp_x, temp_y) && (board[temp_x][temp_y] != 0) && (board[temp_x][temp_y] != color)) {
	found++;
	temp_y += rel_y;
	temp_x += rel_x;
      }
      // If the something different is the original color, than it is a legal move.
      if ((found > 0) && inside_board(temp_x, temp_y) && (board[temp_x][temp_y] == color)) {
	return true;
      }
    }
  }
  return false;
}

void flip_pieces (int x, int y, int color)
{
  // Set the original location as the color placed.
  board[x][y] = color;
  // Go in all eight directions.
  for (int rel_y = -1; rel_y < 2; rel_y++) {
    for (int rel_x = -1; rel_x < 2; rel_x++) {
      int temp_y = y + rel_y;
      int temp_x = x + rel_x;
      // Don't do anything on the spot where the piece is being placed.
      if (rel_y == 0 && rel_x == 0) {
	continue;
      }
      int length = 0;
      // Continue going in a direction where the pieces are opposite colors until something different is reached.
      while (inside_board(temp_x, temp_y) && (board[temp_x][temp_y] != 0) && (board[temp_x][temp_y] != color)) {
	length++;
	temp_y += rel_y;
	temp_x += rel_x;
      }
      // Checking if the direction gone was a legal move.
      if (inside_board(temp_x, temp_y) && (board[temp_x][temp_y] == color)) {
	// If the direction gone was a legal move, then go backwards length times, each time flipping over a piece.
	for (int count = 0; count < length; count++) {
	  temp_x -= rel_x;
	  temp_y -= rel_y;
	  board[temp_x][temp_y] = color;
	}
      }
    }
  }
}

void print_board (int color_to_move)
{
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      if (board[x][y] == black) {
	cout << "B";
      } else if (board[x][y] == white) {
	cout << "W";
      } else if (is_legal_move(x, y, color_to_move)) {
	cout << "L";
      } else {
	cout << board[x][y];
      }
    }
    cout << "\n";
  }
}

int main ()
{
  int x, y;
  int current_color = black;
  setup_board();
  print_board(current_color);
  
  while (true) {
    cout << "\n";
    cout << "Enter an x coordinate and a y coordinate: ";
    cin >> x >> y;

    if (!is_legal_move(x, y, current_color)) {
      cout << "That is not a legal move.";
      continue;
    }
    
    flip_pieces(x, y, current_color);
    if (current_color == black) {
      current_color = white;
    } else if (current_color == white) {
      current_color = black;
    }
    print_board(current_color);
  }
  return 0;
}
