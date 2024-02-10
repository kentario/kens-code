#include <iostream>
#include <SDL2/SDL.h>

int setup_window () {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Could not init SDL: %s\n", SDL_GetError());
    return -1;
  }
  SDL_Window *screen = SDL_CreateWindow ("My application", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 100, 100, 0);
  if (!screen) {
    fprintf (stderr, "Could not create window\n");
    return -2;
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(screen, -1, SDL_RENDERER_SOFTWARE);
  if (!renderer) {
    fprintf (stderr, "Could not create renderer\n");
    return -3;
  }
  
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);

  return 0;
}

int main () {
  if (setup_window() < 0) {
    std::cout << "Big nono happened\n";
    return -1;
  }

  while (1) {
    SDL_Event event;
    while (SDL_WaitEvent(&event) != 1);
    switch (event.type) {
    case SDL_QUIT:
      std::cout << "Quitting.\n";
      return 0;
    }
  }
  
  return 0;
}
