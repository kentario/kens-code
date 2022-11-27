#include <iostream>
#include <SDL2/SDL.h>

using namespace std;

int main ()
{
  SDL_Init(SDL_INIT_EVERYTHING);
  SDL_Window *win = SDL_CreateWindow("title", 30, 15, 600, 500, SDL_WINDOW_SHOWN);
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
    
  bool run = true;
  
  while (run) {
    SDL_Event ev;
    int x = -1;
    int y = -1;
    while (SDL_WaitEvent(&ev) != 1);

    switch (ev.type) {
    case SDL_QUIT:
      cout << "Quitting\n";
      return 0;
      // Keyboard
    case SDL_KEYDOWN:
      cout << SDL_GetKeyName(ev.key.keysym.sym) << " has been pressed\n";
      break;
    case SDL_KEYUP:
      cout << SDL_GetKeyName(ev.key.keysym.sym) << " has been released\n";
      break;
      // Mouse
    case SDL_MOUSEBUTTONDOWN:
      cout << SDL_BUTTON(ev.button.button) << " pressed at x: " << ev.button.x << ", y: " << ev.button.y << "\n";
      break;
    case SDL_MOUSEBUTTONUP:
      cout << SDL_BUTTON(ev.button.button) << " released at x: " << ev.button.x << ", y: " << ev.button.y << "\n";
      x = ev.button.x;
      y = ev.button.y;
      break;
    default:
      continue;
    }
    cout << "here\n";
    if (x >= 0 && y >= 0) {
      SDL_RenderDrawPoint(renderer, x, y);
      SDL_RenderPresent(renderer);
    }
  }
  SDL_DestroyWindow(win);
  SDL_Quit();
}
