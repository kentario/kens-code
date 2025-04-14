#if 0
#include <iostream>

#include <osmium/io/xml_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>

#include <SDL2/SDL.h>

class Road_Map_Drawer : public osmium::handler::Handler {
private:
  static constexpr int SCREEN_WIDTH {800};
  static constexpr int SCREEN_HEIGHT {800};

  SDL_Window *window;
  SDL_Renderer *renderer;

  std::unordered_map<osmium::object_id_type, std::vector<osmium::Location>> roads;
public:
  Road_Map_Drawer () : window {nullptr}, renderer {nullptr} {}

  void render_roads() {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    
    for (const auto &road : roads) {
      const std::vector<osmium::Location> &road_nodes {road.second};
      
      for (size_t i {1}; i < road_nodes.size(); i++) {
	// Draw a line from the previous nodes longitude and latitude, to the current nodes longitude and latitude.
	/*	if (!road_nodes[i - 1].valid()) {
	  std::cout << i << "\n";
	  continue;
	  }*/

	double prev_x {road_nodes[i - 1].lon_without_check()};
	double prev_y {road_nodes[i - 1].lat_without_check()};
	double this_x {road_nodes[i].lon_without_check()};
	double this_y {road_nodes[i].lat_without_check()};
	
	std::cout << "(" << prev_x << ", " << prev_y << "), (" << this_x << ", " << this_y << ")\n";

	//	SDL_RenderDrawLine(renderer, prev_x, prev_y, this_x, this_y);
      }
    }
  }
  
  void draw_map () {
    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("OSM Road Map", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_Event e;
    bool quit {false};

    while (!quit) {
      while (SDL_PollEvent(&e)) {
	if (e.type == SDL_QUIT) {
	  quit = true;
	}
      }

      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
      SDL_RenderClear(renderer);

      render_roads();

      SDL_RenderPresent(renderer);
    }
  }

  void way (osmium::Way &way) {
    // Check if the way is a road.
    if (way.tags().has_key("highway")) {
      std::cout << way.nodes()[0].ref() << "\n";
      std::cout << way.nodes()[0].location().lon_without_check() << ", " << way.nodes()[0].location().lat_without_check() << "\n";
      
      std::vector<osmium::Location> way_node_locations;
      
      // For each node on the road, add its location to the vector of node locations corresponding to this way.
      for (int i {0}; i < way.nodes().size(); i++) {
	way_node_locations.push_back(way.nodes()[i].location());
      }
      /*      for (const NodeRef &node : way.nodes()) {
	way_node_locations.push_back(node.location());
	}*/

      // Then add the vector to a map that goes from the way id to the vector of node locations along that way.
      roads[way.id()] = way_node_locations;
    }
  }
};

int main () {
  try {
    osmium::io::File input_file {"/home/kentario/Downloads/map.osm"};
    osmium::io::Reader reader {input_file};

    Road_Map_Drawer drawer;
    // Calls drawer.way(Way) for each way in the reader.
    osmium::apply(reader, drawer);

    reader.close();
    
    //    drawer.draw_map();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return 0;
}
#endif
/*
#include <iostream>

#include <osmium/io/xml_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

class NodePrinter : public osmium::handler::Handler {
public:
  void way(const osmium::Way& way) {
    std::cout << "Way ID: " << way.id() << std::endl;
    
    for (const auto& node_ref : way.nodes()) {
      osmium::NodeRef node{node_ref};
      std::cout << "    Node ID: " << node.ref()
		<< ", Latitude: " << node.location().lat()
		<< ", Longitude: " << node.location().lon() << std::endl;
    }
  }
};

int main() {
  try {
    osmium::io::File input_file {"/home/kentario/Downloads/map.osm"};  // Replace with your OSM file path
    osmium::io::Reader reader {input_file};

    index_type index;
    location_handler_type location_handler {index};
    
    NodePrinter printer;
    osmium::apply(reader, location_handler, printer);
    
    reader.close();
    
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  
  return 0;
}
//*/

int main () {
  return 0;
}
