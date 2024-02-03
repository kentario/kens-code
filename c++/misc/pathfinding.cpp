#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

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

inline bool operator== (const Point &a, const Point &b) {
  return a.x == b.x && a.y == b.y;
}

inline std::ostream& operator<< (std::ostream &os, const Point &point) {
  os << "(" << point.x << ", " << point.y << ")";
  return os;
}

template <typename T>
inline bool contains (const std::vector<T> &vector, const T &value) {
  auto it = std::find(vector.begin(), vector.end(), value);

  return it != vector.end();
}

struct Node {
  Node *parent {};
  Point location {};
  
  int traveled_distance {-1}; // G cost.
  int target_distance {}; // H cost.
  int f_cost {}; // Sum of G cost and H cost.
  
  // Vector of neighboring node locations, with distance from current node.
  std::vector<std::pair<Point, int>> neighboring_nodes;
};

inline std::ostream& operator<< (std::ostream &os, const Node &node) {
  os << "location: " << node.location <<  ", traveled_distance: " << node.traveled_distance << ", target_distance: " << node.target_distance << ", f_cost: " << node.f_cost << ", neighboring_nodes:\n";
  
  for (const auto &neighboring_node : node.neighboring_nodes) {
    os << "location: " << neighboring_node.first << ", distance: " << neighboring_node.second << "\n";
  }
  
  return os;
}

inline void draw_scaled_rect (SDL_Renderer *renderer, const Point &top_left, const int scale, const SDL_Color &color, const bool outline) {
  SDL_Rect rect {.x = top_left.x * scale, .y = top_left.y * scale, .w = scale, .h = scale};
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
  SDL_RenderFillRect(renderer, &rect);
  if (outline) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &rect);
  }
}

inline int taxicab_distance (const Point &a, const Point &b) {
  return 10 * (std::abs(a.x - b.x) + std::abs(a.y - b.y));
}

// Returns the index of the node with the lowest f_cost. In case of a tie, uses the target_distance.
inline int best_node (const std::vector<Node*> &nodes) {
  if (nodes.empty()) {
    std::cerr << "No new nodes\n";
    exit(EXIT_FAILURE);
  }
  
  int best_node {0};
  for (int i {1}; i < nodes.size(); i++) {
    if ((nodes[i]->f_cost < nodes[best_node]->f_cost) || (nodes[i]->f_cost == nodes[best_node]->f_cost && nodes[i]->target_distance < nodes[best_node]->target_distance)) {
      best_node = i;
    }
  }

  return best_node;
}

class Grid {
private:
  int width;
  int height;
  int scale;
  SDL_Window* window {nullptr};
  SDL_Renderer* renderer {nullptr};
  
  Node** nodes;
  bool** unwalkable;
  
  Point start;
  Point target;

  // purple
  std::vector<Node*> path;
  // blue
  std::vector<Node*> calculated;
  // yellow
  std::vector<Node*> to_be_calculated;
  
  Draw_Type draw_type = Draw_Type::walkable;

  void update_current_node (std::vector<Node*> &to_be_evaluated, std::vector<Node*> &already_evaluated, Node *&current);
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
	nodes[x][y].location = Point{.x = x, .y = y};
	
	// For each node, assign it its neighboring nodes.
	int directions[][2] {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
	for (const auto &direction : directions) {
	  const int rel_x {direction[0]};
	  const int rel_y {direction[1]};
	  
	  const int temp_x {x + rel_x};
	  const int temp_y {y + rel_y};
	  
	  if (temp_x < 0 || temp_x >= width || temp_y < 0 || temp_y >= height) continue;
	  
	  nodes[x][y].neighboring_nodes.push_back(std::make_pair(Point{.x {temp_x}, .y {temp_y}}, 10));
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
      //      std::cout << nodes[descaled_x][descaled_y] << "\n";
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

    // Draw the last calculated path's to be calculated nodes.
    for (auto *&node : to_be_calculated) {
      draw_scaled_rect(renderer, node->location, scale, SDL_Color{.r {255}, .g {255}, .a {255}}, true);
    }
    // Draw the last calculated path's calculated nodes.
    for (auto *&node : calculated) {
      draw_scaled_rect(renderer, node->location, scale, SDL_Color{.b {255}, .a {255}}, true);
      }
    // Draw the last calculated path.
    for (auto *&node : path) {
      draw_scaled_rect(renderer, node->location, scale, SDL_Color{.r {255}, .b {255}, .a {255}}, true);
    }
    
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
      break;
    case SDLK_p:
      // Pathfind from start to end.
      pathfind();
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

  // Will return the path from the target to the start.
  std::vector<Node*> pathfind () {
    std::vector<Node*> to_be_evaluated = {&nodes[start.x][start.y]};
    std::vector<Node*> already_evaluated;
    
    to_be_evaluated[0]->parent = nullptr;
    to_be_evaluated[0]->traveled_distance = 0;
    to_be_evaluated[0]->target_distance = taxicab_distance(to_be_evaluated[0]->location, target);
    to_be_evaluated[0]->f_cost = to_be_evaluated[0]->traveled_distance + to_be_evaluated[0]->target_distance;
    
    for (int i {0}; i < 1000000; i++) {
      // If there are no nodes queued up, then there are no more possible ways to get to the target.
      if (to_be_evaluated.empty()) {
	std::cout << "Target is unreachable\n";
	return std::vector<Node*>{};
      }
      
      //      std::cout << "start of iteration " << i << "\n";
      //      std::cout << "nodes to be evaluated:\n";
      //      for (auto *&node : to_be_evaluated) {
      //      	std::cout << *node << "\n";
      //      }

      // The node in to_be_evaluated with the lowest f_cost in to_be_evaluated. If there is a tie, then choose with the lowest target_distance.
      int best = best_node(to_be_evaluated);
      Node* current = to_be_evaluated[best];

      // Remove current from to_be_evaluated.
      to_be_evaluated.erase(to_be_evaluated.begin() + best);
      already_evaluated.push_back(current);

      // Check if it has reached the target.
      if (current->location == target) {
	std::cout << "target reached\n";
	// The path has been found.
	// Return the path.

	std::vector<Node*> path;
	
	Node *last_node {current};

	while (last_node->parent) {
	  path.push_back(last_node);
	  last_node = last_node->parent;
	}

	this->path = path;
	this->calculated = already_evaluated;
	this->to_be_calculated = to_be_evaluated;
	
	return path;
      }
      
      //      std::cout << "neighboring nodes:\n";
      update_current_node(to_be_evaluated, already_evaluated, current);
      //      std::cout << "\n";
    }

    std::cerr << "Maximum deapth reached\n";
    exit(EXIT_FAILURE);
  }
};

void Grid::update_current_node (std::vector<Node*> &to_be_evaluated, std::vector<Node*> &already_evaluated, Node *&current) {
  //  std::cout << *current << "\n";
  for (auto &neighboring_node : current->neighboring_nodes) {
    Node *neighboring_node_p {&nodes[neighboring_node.first.x][neighboring_node.first.y]};
    //	std::cout << *neighboring_node_p << "\n";
    const bool neighboring_node_unwalkable {unwalkable[neighboring_node.first.x][neighboring_node.first.y]};
    //	std::cout << "node is walkable: " << (!neighboring_node_unwalkable ? "true" : "false") << "\n";
    
    if (neighboring_node_unwalkable || contains(already_evaluated, neighboring_node_p)) {
      continue;
    }

    //  If the neighboring node has not been evaluated, or the traveled distance is closer than some previously computed traveled distance.
    if (!contains(to_be_evaluated, neighboring_node_p) || (current->traveled_distance + neighboring_node.second) < neighboring_node_p->traveled_distance) {
      // Calculate the traveled distance, and the target distance.
      neighboring_node_p->traveled_distance = current->traveled_distance + neighboring_node.second;
      neighboring_node_p->target_distance = taxicab_distance(target, neighboring_node_p->location);
      // The f_cost is the sum of those 2 distances.
      neighboring_node_p->f_cost = neighboring_node_p->traveled_distance + neighboring_node_p->target_distance;
      // The path to the neighboring node came through the current node.
      neighboring_node_p->parent = current;
      if (!contains(to_be_evaluated, neighboring_node_p)) {
	to_be_evaluated.push_back(neighboring_node_p);
      }
    }
  }
}

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
  
  Grid grids[] {Grid(20, 20, 75, "test_grid")};

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
