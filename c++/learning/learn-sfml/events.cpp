#include <vector>

#include <SFML/Graphics.hpp>

int main (int argc, char *argv[]) {
  sf::RenderWindow window{sf::VideoMode {{200, 200}}, "Hello World", sf::Style::Titlebar | sf::Style::Close};

  std::vector<sf::CircleShape> circles {};
  circles.reserve(1000);

  /*
   * Using sf::WindowBase::handleEvents for event visitation.
   * https://www.sfml-dev.org/documentation/3.0.0/classsf_1_1WindowBase.html#ad86ae79ff4e2da25af1ca3cd06f79557
   */
  
  const auto on_close = [&window](const sf::Event::Closed&) {
    window.close();
  };

  const auto place_circle = [&window, &circles](const sf::Vector2f &position, const sf::Color color = sf::Color::Black) {
    sf::CircleShape circle {};
    circle.setRadius(5);
    circle.setPosition(position);
    circle.setFillColor(color);
    circles.push_back(circle);
  };

  // Only run the program while the window is open.
  while (window.isOpen()) {
    // Handle all new events since the last iteration of the loop.
    // Using a named lambda with on_close because I want that to be the same in most cases.
    // Using an inline lambda for the MouseMoved and MouseButtonPressed lambdas because in different cases these events might result in different outcomes.
    window.handleEvents(on_close,
			[&window, &circles, &place_circle](const sf::Event::MouseMoved move) {
			  // If left click is dragged, then place a circle along the path of the mouse.
			  if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
			    place_circle(sf::Vector2f {move.position}, sf::Color {25, 140, 160});
			  }
			  // If right click is dragged, then place a black circle so it it as if it is erasing the previous circles.
			  // No need to worry about circles layering on top of each other because they will be drawn in the same order that they were created.
			  else if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
			    place_circle(sf::Vector2f {move.position});
			  }
			},
			[&window, &circles, &place_circle](const sf::Event::MouseButtonPressed button_press) {
			  // Also place a circle on a left click.
			  switch (button_press.button) {
			  case sf::Mouse::Button::Left:
			    place_circle(sf::Vector2f {button_press.position}, sf::Color {25, 140, 160});
			    break;
			  case sf::Mouse::Button::Right:
			    place_circle(sf::Vector2f {button_press.position});
			    break;
			  }
			}
			);
    
    // Clear old frame
    window.clear(sf::Color::Black);
    
    // Draw Stuff
    for (const auto &circle : circles) {
      window.draw(circle);
    }
    
    // Display the drawn stuff.
    window.display();
  }
  
  return 0;
}
