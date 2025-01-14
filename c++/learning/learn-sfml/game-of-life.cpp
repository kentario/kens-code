#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <SFML/Graphics.hpp>

using namespace std::chrono_literals;

template <typename T>
std::string to_string (const sf::Vector2<T> &vec) {
  return std::to_string(vec.x) + " " + std::to_string(vec.y);
}

bool in_screen (const std::vector<std::vector<bool>> &cells, const sf::Vector2<size_t> &cell) {
  return cell.y < cells.size() && cell.x < cells[0].size();
}

sf::Vector2<size_t> pixel_to_cell (const sf::Vector2<int> &position, const size_t cell_width) {
  return sf::Vector2 {
    static_cast<size_t>(std::max(0, position.x))/cell_width,
    static_cast<size_t>(std::max(0, position.y))/cell_width
  };
}

unsigned int count_adjacent (const std::vector<std::vector<bool>> &cells, const sf::Vector2<size_t> &cell) {
  unsigned int count{0};

  for (int rel_y{-1}; rel_y <= 1; rel_y++) {
    for (int rel_x{-1}; rel_x <= 1; rel_x++) {
      // Skip the current cell.
      if (!rel_y && !rel_x) continue;

      // Converting to size_t makes negative numbers into very big numbers (which are still seen as invalid by in_screen).
      const sf::Vector2<size_t> new_cell{
	static_cast<size_t>(cell.x + rel_x),
	static_cast<size_t>(cell.y + rel_y)
      };
      if (in_screen(cells, new_cell) && cells[new_cell.y][new_cell.x]) {
	count++;
      }
    }
  }

  return count;
}

void update_game (std::vector<std::vector<bool>> &cells) {
  // Writing the new cell statuses to a separate vector so that it doesn't mess with counting the living adjacent cells.
  std::vector<std::vector<bool>> temp(cells.size(), std::vector<bool>(cells[0].size(), false));

  for (size_t y{0}; y < cells.size(); y++) {
    for (size_t x{0}; x < cells[0].size(); x++) {
      const unsigned int adjacent{count_adjacent(cells, {x, y})};
      // If there are 3 adjacent, then the cell is guaranteed to stay/become alive.
      // A cell that is already alive can stay alive if it has 2 adjacent living cells.
      temp[y][x] = adjacent == 3 || (cells[y][x] && adjacent == 2);
    }
  }

  cells = std::move(temp);
}

int main (int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "usage: " << argv[0] << " [width] [height] [cell width]\n";
    return 1;
  }

  const size_t width     {std::stoul(argv[1])};
  const size_t height    {std::stoul(argv[2])};
  const size_t cell_width{std::stoul(argv[3])};

  std::vector<std::vector<bool>> cells(height, std::vector<bool>(width, false));

  sf::RenderWindow window{
    sf::VideoMode {{
	static_cast<unsigned int>(width * cell_width),
	static_cast<unsigned int>(height * cell_width)
      }},
    "Game of Life",
    sf::Style::Titlebar | sf::Style::Close
  };

  const auto on_close = [&window](const sf::Event::Closed&) {
    window.close();
  };

  // True for drawing, false for erasing.
  bool drawing{true};
  // True while the game is playing, false if it is paused.
  bool updating{true};
  // The delay between updates in seconds.
  std::chrono::duration delay{0.1s};

  auto last_update = std::chrono::steady_clock::now();
  while (window.isOpen()) {
    // Some keyboard key presses have a meaning.
    window.handleEvents(on_close,
			[&window, &updating, &drawing, &delay, &cells, width](const sf::Event::KeyPressed &key_press) {
			  switch (key_press.code) {
			  case sf::Keyboard::Key::P:
			    updating = false;
			    break;
			  case sf::Keyboard::Key::U:
			    updating = true;
			    break;
			  case sf::Keyboard::Key::Space:
			    updating = !updating;
			    break;
			    // Speed up and slow down the updating of the game, with a minimum delay of 0.1s.
			  case sf::Keyboard::Key::Up:
			    delay += 0.1s;
			    break;
			  case sf::Keyboard::Key::Down:
			    delay = std::max(0.1s, delay - 0.1s);
			    break;
			  case sf::Keyboard::Key::Backspace:
			    drawing = false;
			    break;
			  case sf::Keyboard::Key::Enter:
			    drawing = true;
			    break;
			    // Clear the screen.
			  case sf::Keyboard::Key::Delete:
			    std::fill(cells.begin(), cells.end(), std::vector<bool>(width, false));
			    break;
			  }
			},
			[&window, &cells, cell_width, drawing](const sf::Event::MouseMoved &mouse_move) {
			  if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
			    const sf::Vector2 cell{pixel_to_cell(mouse_move.position, cell_width)};
			    // Dragging the cursor off the window while holding down a mouse button will result in pixel positions that are not within the window bounds.
			    if (in_screen(cells, cell)) {
			      cells[cell.y][cell.x] = drawing;
			    }
			  }
			},
			[&window, &cells, cell_width, drawing](const sf::Event::MouseButtonPressed &mouse_press) {
			  const sf::Vector2 cell{pixel_to_cell(mouse_press.position, cell_width)};
			  cells[cell.y][cell.x] = drawing;
			});

    const auto current_time = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update);

    if (elapsed_time >= delay && updating) {
      update_game(cells);
      last_update = std::chrono::steady_clock::now();
    }

    // Clearing screen.
    window.clear(sf::Color::Black);
    // Drawing cells.
    for (size_t y{0}; y < cells.size(); y++) {
      for (size_t x{0}; x < cells[0].size(); x++) {
	sf::RectangleShape cell{{
	    static_cast<float>(cell_width),
	    static_cast<float>(cell_width)
	  }};
	cell.setPosition({
	    static_cast<float>(x * cell_width),
	    static_cast<float>(y * cell_width)
	  });
	cell.setFillColor(sf::Color::Black);

	if (cells[y][x]) {
	  cell.setFillColor(sf::Color::White);
	} else if (!updating) {
	  // If the game is paused, draw an outline around empty cells.
	  cell.setOutlineThickness(-1);
	  cell.setOutlineColor(sf::Color::White);
	}
	window.draw(cell);
      }
    }
    // Displaying cells.
    window.display();
  }

  return 0;
}
