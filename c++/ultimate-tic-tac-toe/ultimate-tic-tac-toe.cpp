#include <iostream>
#include <random>
#include <exception>
#include <memory>
#include <array>
#include <vector>

#include "heuristic.hpp"
#include "game.hpp"
#include "bots.hpp"
#include "simulate.hpp"
#include "benchmark.hpp"

int main () {
  try {
    std::mt19937 rng {};
    // std::array<Board, 100'000> boards {};
    // for (Board &b : boards) {
    //   b = random_position(rng, false);
    // }
    // save_positions("positions-100k", boards.data(), boards.size(), false);
    
    //TODO SOMEWHERE ELSE
    // make a function to check if the current position will just result in a draw
    // todo
    // pregenerate legal moves for all possible subboards.
    
    std::vector boards100k {load_positions("positions-100k", 100'000)};
    
    std::array<Board, 1000> boards {};
    for (int i {0}; i < 1'000; i++) {
      boards[i] = random_position(rng, false, 10, 81);
    }
    std::sort(boards.begin(), boards.end(), [](const Board &a, const Board &b) {
      return a.count_total_empty_squares() > b.count_total_empty_squares();
    });

    Minimax m {"minimax", 5, Eval_Params {}, &heur2, rng()};
    Negamax n {"negamax", 5, Eval_Params {}, &heur2, rng()};

    //    std::cout << benchmark_bot_move_generation(m, boards100k);
    //    std::cout << benchmark_bot_move_generation(n, boards100k);
    
    /*4
    Minimax_Full b {"test", 1, Eval_Params {}, &heur1, true, rng()};
    //for checking minimaxfull efficiency/validity.
    for (int i {999}; i >= 0; i--) {
      std::cout << i << '\n';
      std::cout << role(boards[i].next_player()) << " to play\n";
      std::cout << boards[i] << "\n";
      std::cout << "there are " << boards[i].count_total_empty_squares() << " empty squares in the board\n";
      auto m = b(boards[i]);
      std::cout << m << "\n\n\n";
      if (!boards[i].is_legal(m)) {
	std::cerr << "super bad stuff";
	std::cerr << i << " " << m;
	return EXIT_FAILURE;
      }
      }*/

    //    std::cout << benchmark_find_legal_moves(boards100k) << '\n';

    Board::pre_generate_legal_moves(false);
    // Check if most of the moves are correct.
    std::cout << "here\n";
    for (int i {0}; i < 100; i++) {
      Board board = boards100k[i];
      std::cout << board << '\n';

      if (board.legal_moves_new() != board.legal_moves()) {
	std::cout << "new: ";
	for (const auto move : board.legal_moves_new()) std::cout << move << ' ';
	std::cout << '\n';
	std::cout << "old: ";
	for (const auto move : board.legal_moves()) std::cout << move << ' ';
	std::cout << '\n';
	return EXIT_FAILURE;
      }

      std::cout << "\n\n";
    }

    std::cout << "success\n";

  } catch (const std::exception &e) {
    std::cerr << e.what();
    
    return EXIT_FAILURE;
  }
  
  return 0;
}
