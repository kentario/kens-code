#include <iostream>
#include <vector>
#include <utility>
#include <SDL2/SDL.h>

enum class Draw_Type {
  walkable,
  unwalkable,
  start,
  target,
  invert
};

struct Point {
  int x {};
  int y {};
};

struct Node {
  int traveled_distance {}; // G cost.
  int target_distance {}; // H cost.
  int f_cost {}; // Sum of G cost and H cost.

  // Vector of neighboring nodes, with distance from current node.
  std::vector<std::pair<Node*, int>> neighboring_nodes;
};

inline void draw_scaled_rect (SDL_Renderer *renderer, const Point &top_left, const int scale, const SDL_Color &color, const bool outline) {
  SDL_Rect rect {.x = top_left.x * scale, .y = top_left.y * scale, .w = scale, .h = scale};
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
  SDL_RenderFillRect(renderer, &rect);
  if (outline) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &rect);
  }
}

class Grid {
private:
  int width;
  int height;
  int scale;
  SDL_Window* window {nullptr};
  SDL_Renderer* renderer {nullptr};

  //  std::vector<std::vector<Node>> nodes;
  Node** nodes;

  bool** unwalkable;

  // Using unsigned char because std::vector<bool> does weird things.
  //  std::vector<std::vector<unsigned char>> unwalkable;

  Point start;
  Point target;

  Draw_Type draw_type = Draw_Type::walkable;
public:
  Grid (int width, int height, int scale, const char *name = "Grid") :
    width{width}, height{height}, scale{scale} {
    // Initialize SDL.
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
      std::cerr << "SDL initialization failed: " << SDL_GetError() << "\n";
      exit(EXIT_FAILURE);
    }
    
    // Create SDL window.
    window = SDL_CreateWindow(name, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width * scale, height * scale, SDL_WINDOW_SHOWN);
    if (!window) {
      std::cerr << "Window creation failed: " << SDL_GetError() << "\n";
      exit(EXIT_FAILURE);
    }
    
    // Create renderer.
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
      std::cerr << "Renderer creation failed: " << SDL_GetError() << "\n";
      exit(EXIT_FAILURE);
    }
    // Allows the drawing of transparent colors, which will blend with those below them.
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    nodes = new Node*[width];
    unwalkable = new bool*[width];
    
    for (int x {0}; x < width; x++) {
      nodes[x] = new Node[height]{};
      unwalkable[x] = new bool[height]{};
      
      for (int y {0}; y < height; y++) {
	// For each node, assign it its neighboring nodes.
	int directions[][2] {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
	for (const auto &direction : directions) {
	  const int rel_x {direction[0]};
	  const int rel_y {direction[1]};

	  const int temp_x {x + rel_x};
	  const int temp_y {y + rel_y};
	  
	  if (temp_x < 0 || temp_x >= width || temp_y < 0 || temp_y >= height) continue;
	  
	  nodes[x][y].neighboring_nodes.push_back(std::make_pair(&nodes[temp_x][temp_y], 10));
	}
      }
    }
    
    update_screen();
  }
  
  ~Grid () {
    for (int x {0}; x < width; x++) {
      delete[] nodes[x];
      delete[] unwalkable[x];
    }
    delete[] nodes;
    delete[] unwalkable;
    
    if (renderer) {
      SDL_DestroyRenderer(renderer);
    }
    if (window) {
      SDL_DestroyWindow(window);
    }
    SDL_Quit();
  }
  
  SDL_Window* get_window () {return window;}
  
  void handle_mouse_event (SDL_Event &event) {
    int descaled_x {event.button.x/scale};
    int descaled_y {event.button.y/scale};
	
    // Find if it was a click or a drag.
    switch (event.type) {
    case SDL_MOUSEBUTTONUP:
      if (draw_type == Draw_Type::invert) {
	// A click during invert mode will inver the positions of start and finish, if the start or finish were clicked.
	if ((descaled_x == start.x && descaled_y == start.y) || (descaled_x == target.x && descaled_y == target.y)) {
	  Point temp_start {start};
	  start.x = target.x;
	  start.y = target.y;
	  target.x = temp_start.x;
	  target.y = temp_start.y;
	} else {
	  // Otherwise it happened on a walkable or unwalkable, so invert that square.
	  unwalkable[descaled_x][descaled_y] = !unwalkable[descaled_x][descaled_y];
	}
      }
      break;
    case SDL_MOUSEMOTION:
      // Find out what key was dragged (if any).
      if (event.motion.state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
	//	std::cout << "start: " << start.x << " " << start.y << " target: " << target.x << " " << target.y << std::endl;
	//	std::cout << descaled_x << " " << descaled_y << " unwalkable: " << unwalkable[descaled_x][descaled_y] << std::endl;
	switch (draw_type) {
	case Draw_Type::walkable:
	  unwalkable[descaled_x][descaled_y] = false;
	  break;
	case Draw_Type::unwalkable:
	  unwalkable[descaled_x][descaled_y] = true;
	  break;
	case Draw_Type::start:
	  start.x = descaled_x;
	  start.y = descaled_y;
	  break;
	case Draw_Type::target:
	  target.x = descaled_x;
	  target.y = descaled_y;
	  break;
	}
	//      } else if (event.motion.state & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
	//	// Nothing implemented yet.
	//      } else if (event.motion.state & SDL_BUTTON(SDL_BUTTON_MIDDLE)) {
	//	// Nothing implemented yet.
      }
    }
    
    // The start and end are always walkable.
    unwalkable[start.x][start.y] = false;
    unwalkable[target.x][target.y] = false;
  }

  void update_screen () {
    // Clear the screen first.
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Draw the start and target squares.
    draw_scaled_rect(renderer, Point{.x {start.x}, .y {start.y}}, scale, SDL_Color{.g {255}, .a {255}}, true);
    draw_scaled_rect(renderer, Point{.x {target.x}, .y {target.y}}, scale, SDL_Color{.r {255}, .a {255}}, true);
 
    // Draw the unwalkable squares.
    for (int x {0}; x < width; x++) {
      for (int y {0}; y < height; y++) {
	// If the square is unwalkable, then it should be visible, but if it is walkable, then it should see through.
	uint8_t alpha {static_cast<uint8_t>(unwalkable[x][y] ? 255 : 0)};
	
	draw_scaled_rect(renderer, Point{.x {x}, .y {y}}, scale, SDL_Color{.r {255}, .g {255}, .b {255}, .a {alpha}}, true);
      }
    }
    
    SDL_RenderPresent(renderer);
  }
  
  void handle_key_press (SDL_Event &event) {
    // Find the type of keypress.
    switch (event.key.keysym.sym) {
    case SDLK_BACKSPACE:
    case SDLK_w:
    case SDLK_0:
      draw_type = Draw_Type::walkable;
      break;
    case SDLK_u:
    case SDLK_1:
      draw_type = Draw_Type::unwalkable;
      break;
    case SDLK_2:
    case SDLK_s:
      draw_type = Draw_Type::start;
      break;
    case SDLK_e:
    case SDLK_3:
    case SDLK_t:
      draw_type = Draw_Type::target;
      break;
    case SDLK_4:
    case SDLK_i:
      draw_type = Draw_Type::invert;
    }
  }

  void handle_events (SDL_Event &event) {
    switch (event.type) {
    case SDL_WINDOWEVENT:
      // Check if the event is trying to close the window.
      if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
	SDL_DestroyWindow(window);
	window = nullptr;
	return;
      }
      break;
    case SDL_KEYDOWN:
      handle_key_press(event);
      break;
    case SDL_MOUSEBUTTONUP:
    case SDL_MOUSEMOTION:
      handle_mouse_event(event);
      break;
    case SDL_QUIT:
      exit(0);
    }

    update_screen();
  }
};

void printEventDetails (const SDL_Event& event) {
  switch (event.type) {
  case SDL_KEYDOWN:
    std::cout << "Key pressed: " << event.key.keysym.sym << std::endl;
    break;
  case SDL_KEYUP:
    std::cout << "Key released: " << event.key.keysym.sym << std::endl;
    break;
  case SDL_MOUSEBUTTONDOWN:
    std::cout << "Mouse button pressed at: (" << event.button.x << ", " << event.button.y << ")" << std::endl;
    break;
  case SDL_MOUSEBUTTONUP:
    std::cout << "Mouse button released at: (" << event.button.x << ", " << event.button.y << ")" << std::endl;
    break;
  case SDL_MOUSEMOTION:
    std::cout << "Mouse moved to: (" << event.motion.x << ", " << event.motion.y << ")" << std::endl;
    break;
  case SDL_WINDOWEVENT:
    switch (event.window.event) {
    case SDL_WINDOWEVENT_CLOSE:
      std::cout << "Window closed" << std::endl;
      break;
      // Other window events...
    default:
      break;
    }
    break;
  case SDL_QUIT:
    std::cout << "Quit event received" << std::endl;
    break;
    // Other event types...
  default:
    break;
  }
}

int main () {
  std::cout << __cplusplus << "\n";
  
  Grid grids[] {Grid(10, 10, 50, "test_grid"), Grid(15, 15, 25, "other_grid")};

  bool windows_remain {true};
  
  SDL_Event event;
  while (windows_remain) {
    while (SDL_PollEvent(&event)) {
      windows_remain = false;
      //      printEventDetails(event);
      // Find the window that corresponds to the event's window.
      for (Grid &grid : grids) {
	if (!grid.get_window()) continue;
	windows_remain = true;
	
	if (event.window.windowID == SDL_GetWindowID(grid.get_window())) {
	  grid.handle_events(event);
	  break;
	}
      }
    }
  }
  
  return 0;
}
