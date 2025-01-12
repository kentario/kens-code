#include <iostream>
#include <vector>
#include <chrono>

#include <SFML/Graphics.hpp>

using namespace std::chrono_literals;

template <typename T>
std::string to_string (const sf::Vector2<T> &vec) {
  return std::to_string(vec.x) + " " + std::to_string(vec.y);
}

bool in_screen (const std::vector<std::vector<bool>> &cells, const sf::Vector2i &cell) {
  return cell.y >= 0 && cell.y < cells.size() && cell.x >= 0 && cell.x < cells[0].size();
}

unsigned int count_adjacent (const std::vector<std::vector<bool>> &cells, const sf::Vector2i &cell) {
  unsigned int count{0};
  
  for (int rel_y{-1}; rel_y <= 1; rel_y++) {
    for (int rel_x{-1}; rel_x <= 1; rel_x++) {
      // Skip the current cell.
      if (!rel_y && !rel_x) continue;
      
      const sf::Vector2i new_cell{cell + sf::Vector2i {rel_x, rel_y}};
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

sf::Vector2i pixel_to_cell (const sf::Vector2i &position, const size_t cell_width) {
  return sf::Vector2i {position.x/cell_width, position.y/cell_width};
}

int main (int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Correct usage: " << argv[0] << " <width> " << "<height>\n";
    return 1;
  }

  const size_t width{std::stoul(argv[1])};
  const size_t height{std::stoul(argv[2])};
  constexpr size_t cell_width {100};
  
  std::vector<std::vector<bool>> cells(height, std::vector<bool>(width, false));
  
  sf::RenderWindow window{sf::VideoMode {{width * cell_width, height * cell_width}}, "Game of Life", sf::Style::Titlebar | sf::Style::Close};

  const auto on_close = [&window](const sf::Event::Closed&) {
    window.close();
  };

  // True for drawing, false for erasing.
  bool drawing{true};
  // True while the game is playing, false if it is paused.
  bool updating{true};
  // The delay between updates in milliseconds.
  std::chrono::duration delay{1s};
  
  auto last_update = std::chrono::steady_clock::now();
  while (window.isOpen()) {
    // Some keyboard key presses have a meaning.
    window.handleEvents(on_close,
			[&window, &updating, &drawing, &delay](const sf::Event::KeyPressed &key_press) {
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
			  }
			},
			[&window, &cells, drawing](const sf::Event::MouseMoved &mouse_move) {
			  if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
			    const sf::Vector2i cell{pixel_to_cell(mouse_move.position, cell_width)};
			    // Dragging the cursor off the window while holding down a mouse button will result in pixel positions that are not within the window bounds.
			    if (in_screen(cells, cell)) {
			      cells[cell.y][cell.x] = drawing;
			    }
			  }
			},
			[&window, &cells, drawing](const sf::Event::MouseButtonPressed &mouse_press) {
			  const sf::Vector2u cell{pixel_to_cell(mouse_press.position, cell_width)};
			  cells[cell.y][cell.x] = drawing;
			});

    const auto current_time = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update);

    if (elapsed_time >= delay && updating) {
      update_game(cells);
      last_update = std::chrono::steady_clock::now();
      std::cout << "update\n";
    }
    window.clear(sf::Color::Black);
    
    for (size_t y{0}; y < cells.size(); y++) {
      for (size_t x{0}; x < cells[0].size(); x++) {
	sf::RectangleShape cell{{cell_width, cell_width}};
	cell.setPosition({static_cast<float>(x * cell_width), static_cast<float>(y * cell_width)});
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
    window.display();
  }
  
  return 0;
}
