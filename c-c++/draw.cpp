#include <iostream>
#include <SDL2/SDL.h>
#include <string.h>

using namespace std;

void draw_scaled_point (SDL_Renderer *renderer, int scale, int x, int y, int type) {
  // Set color to white.
  switch (type) {
  case 0:
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    break;
  case 1:
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    break;
  }
  for (int rel_y = 0; rel_y < scale; rel_y++) {
    for (int rel_x = 0; rel_x < scale; rel_x++) {
      SDL_RenderDrawPoint(renderer, (x * scale) + rel_x, (y * scale) + rel_y);
    }
  }
}

int main () {
  int scale = 10;
  int width = 16, height = 16;
  int type[width/scale][height/scale];
  bzero(type, sizeof(type));
  
  // Set up SDL2.
  SDL_Init(SDL_INIT_EVERYTHING);
  // First two numbers are the position of the window on the screen, second two numbers are the size of the screen.
  SDL_Window *win = SDL_CreateWindow("Numbers", 30, 15, width * scale, height * scale, SDL_WINDOW_SHOWN);
  SDL_Surface *screen = SDL_GetWindowSurface(win);
  SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);

  // Change SDL drawing color.
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  // Fill the screen with the color.
  SDL_RenderClear(renderer);
  // Draw anything that has not been rendered yet. Similar to refreshing the screen.
  SDL_RenderPresent(renderer);
  // Set color to white.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  
  while (1) {
    SDL_Event event;
    while (SDL_WaitEvent(&event) != 1);
    switch (event.type) {
    case SDL_QUIT:
      cout << "Quitting\n";
      return 0;
    case SDL_MOUSEMOTION:
      //cout << SDL_BUTTON(event.button.button) << "\n";
      // Scaling down the point.
      int descaled_x = (event.button.x)/scale;
      int descaled_y = (event.button.y)/scale;
      if (SDL_BUTTON(event.button.button) == 1) {
	type[descaled_x][descaled_y] = 1;
	draw_scaled_point(renderer, scale, descaled_x, descaled_y, 1);
      } else if (SDL_BUTTON(event.button.button) == 8) {
	type[descaled_x][descaled_y] = 0;
	draw_scaled_point(renderer, scale, descaled_x, descaled_y, 0);
      }
      SDL_RenderPresent(renderer);
    }
  }
}
