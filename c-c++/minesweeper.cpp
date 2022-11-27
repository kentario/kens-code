#include <iostream>
#include <SDL2/SDL.h>
#include <cstring>
#include <cstdlib>
#include <time.h>

using namespace std;

const int width = 10;
const int height = 10;

void around_zero (int x, int y, bool is_visible[][height], int adjacent[][height])
{
  for (int rel_y = -1; rel_y < 2; rel_y++) {
    for (int rel_x = -1; rel_x <2; rel_x++) {
      // Making sure to not check the original coordiatnes to prevent infinite loops.
      if (rel_x != 0 || rel_y != 0) {
	int this_x = x + rel_x;
	int this_y = y + rel_y;
	if (this_x < 0) {
	  this_x = 0;
	} else if (this_y < 0) {
	  this_y = 0;
	} else if (this_x >= width) {
	  this_x = width - 1;
	} else if (this_y >= height) {
	  this_y = height - 1;
	}
	// Making sure that only a square that is not visible is called to prevent infinite loops.
	if (adjacent[this_x][this_y] == 0 && is_visible[this_x][this_y] == false) {
	  is_visible[this_x][this_y] = true;
	  around_zero(this_x, this_y, is_visible, adjacent);
	}
	is_visible[this_x][this_y] = true;
      }
    }
  }
}

int main ()
{
  int bomb_count = 15;
  int scale = 50;
  
  bool bombs[width][height];
  bool is_visible[width][height];
  int adjacent[width][height];

  // Setting up SDL2. 
  SDL_Init(SDL_INIT_EVERYTHING);
  // First two numbers are the position of the window on the screen, second two numbers are the size of the screen.
  SDL_Window *win = SDL_CreateWindow("Minesweeper", 30, 15, width * scale, height * scale, SDL_WINDOW_SHOWN);
  SDL_Surface *screen = SDL_GetWindowSurface(win);
  SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
  // Change SDL drawing color.
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  // Fill the screen with the color.
  SDL_RenderClear(renderer);
  // Draw anything that has not been rendered yet. Similar to refreshing the screen.
  SDL_RenderPresent(renderer);
 
  // Settings the seed of the random numbers to the current time.
  srand(time(0));
  
  // Settings all arrays to default settings.
  memset(bombs, false, sizeof(bombs));
  memset(is_visible, false, sizeof(is_visible));
  memset(adjacent, 0, sizeof(adjacent));
  
  // Placing bombs.
  int placed_bombs = 0;
  while (placed_bombs < bomb_count) {
    int current_rand_x = rand()%width;
    int current_rand_y = rand()%height;
    if (!bombs[current_rand_x][current_rand_y]) {
      bombs[current_rand_x][current_rand_y] = true;
      placed_bombs++;
    }
  }
  
  // Counting adjacent bombs.
  for (int current_y = 0; current_y < height; current_y++) {
    for (int current_x = 0; current_x < width; current_x++) {
      for (int rel_y = -1; rel_y < 2; rel_y++) {
	for (int rel_x = -1; rel_x < 2; rel_x++) {
	  int this_x = current_x + rel_x;
	  int this_y = current_y + rel_y;
	  // Checking if the square being checked is outside of the array.
	  if (!(this_x < 0 || this_y < 0 || this_x >= width || this_y >= height)) {
	    // Checking if square contains a bomb.
	    if (bombs[this_x][this_y]) {
	      adjacent[current_x][current_y]++;
	    }
	  }
	}
      }
    }
  }
  
  while (1) {
    int squares_visible = 0;
    int input_x, input_y;
    
    // Getting user input.
    SDL_Event event;
    while (SDL_WaitEvent(&event) != 1);
    switch (event.type) {
    case SDL_QUIT:
      cout << "Quitting\n";
      return 0;
    case SDL_MOUSEBUTTONUP:
      input_x = (event.button.x)/scale;
      input_y = (event.button.y)/scale;
      cout << input_x << ", " << input_y << "\n";
    default:
      continue;
    }
    
    //    cout << "Enter the horizontal coordinate: ";
    //    cin >> input_x;
    //    cout << "Enter the vertical coordinate: ";
    //    cin >> input_y;
    //    is_visible[input_x][input_y] = true;

    // If selected square is a zero, then opening all the squares around it.
    if (adjacent[input_x][input_y] == 0) {
      around_zero(input_x, input_y, is_visible, adjacent);
    }
    
    // Printing the field.
    for (int current_y = 0; current_y < height; current_y++) {
      for (int current_x = 0; current_x < width; current_x++) {
	if (!is_visible[current_x][current_y]) {
	  cout << ".";
	} else if (bombs[current_x][current_y]) {
	  cout << "*";
	} else {
	  cout << adjacent[current_x][current_y];
	}
      }
      cout << "\n";
    }
    
    // Checking if the game should continue.
    for (int current_y = 0; current_y < height; current_y++) {
      for (int current_x = 0; current_x < width; current_x++) {
	// Checking if any bombs are visible.
	if (bombs[current_x][current_y] && is_visible[current_x][current_y]) {
	  cout << "Lose\n";
	  return 0;
	}
	// Couting the number of visible squares.
	if (is_visible[current_x][current_y]) {
	  squares_visible++;
	}
	// Checking if all non-bomb squares are visible.
	if (squares_visible + bomb_count == width * height) {
	  cout << "Win\n";
	  return 0;
	}
      }
    }
  }
  return 0;
}
