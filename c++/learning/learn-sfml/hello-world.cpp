#include <iostream>

#include <SFML/Graphics.hpp>

int main (int argc, char *argv[]) {
  sf::RenderWindow window{sf::VideoMode {200, 200}, "Hello World"};
  sf::Event ev;

  while (window.isOpen()) {
    while (window.pollEvent(ev)) {
      switch (ev.type) {
      case sf::Event::Closed:
	window.close();
	break;
      case sf::Event::KeyPressed:
	std::cout << ev.key.code << '\n';
	break;
      }
    }

    // Update

    // Clear old frame
    window.clear();
    
    // Draw Stuff
    
    // Render the stuff drawn into the screen.
    window.display();
  }
  
  return 0;
}
