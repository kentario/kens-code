#include <iostream>

#include "adder.hpp"

#include <SFML/Graphics.hpp>

int main () {
  std::cout << "Hello CMake!\n";
  std::cout << "3 + 5 = " << add(3, 5) << '\n';
  sf::RenderWindow window{sf::VideoMode {{200, 200}}, "Hello World", sf::Style::Titlebar | sf::Style::Close};

  // Only run the program while the window is open.
  while (window.isOpen()) {
    // Check all triggered events since the last iteration of the loop.
    while (const std::optional event {window.pollEvent()}) {
      if (event->is<sf::Event::Closed>()) {
	window.close();
      }
    }

    // Clear old frame.
    // Not needed because nothing is being drawn.
    window.clear(sf::Color::Black);
    
    // Draw Stuff.
    
    // Display the drawn stuff.
    window.display();
  }
  
  return 0;
}
