#include <iostream>

#include <SDL2/SDL.h>

int main()
{
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Could not init SDL: %s\n", SDL_GetError());
    return 1;
  }
  SDL_Window *screen = SDL_CreateWindow ("My application", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 100, 100, 0);
  if (!screen) {
    fprintf (stderr, "Could not create window\n");
    return 1;
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(screen, -1, SDL_RENDERER_SOFTWARE);
  if (!renderer) {
    fprintf (stderr, "Could not create renderer\n");
    return 1;
  }
  
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);

  for (int x = 0; x < 10; x ++) {
    for (int i = 0; i < 10; i ++) {
      SDL_RenderDrawPoint(renderer, 50, x);
    }
  }

  SDL_RenderPresent(renderer);
  SDL_Delay(1000);
  
  SDL_DestroyWindow(screen);
  SDL_Quit();
  return 0;
}
