#include <iostream>
#include <random>
#include <exception>
#include <memory>
#include <algorithm>
#include <array>
#include <vector>
#include <filesystem>
#include <string>
#include <string_view>

#include "bots.hpp"
#include "heuristic.hpp"
#include "game.hpp"
#include "simulate.hpp"
#include "benchmark.hpp"
#include "tune.hpp"

std::string trim_space (const std::string &str) {
  size_t i {0};
  for (; str[i] == ' ' && i < str.size(); i++);
  return str.substr(i);
}

template <size_t max_depth, Heuristic eval>
void play_game_with_player (Negamax<max_depth, eval> *bot, const size_t full_search_threshold, const Player player, Board &board) {
  for (int i {0}; !board.terminal(); i++) {
    std::cout << board << '\n';
    Move move {};
    std::cout << board.next_player() << " to play\n";
    if (board.next_player() == player) {
      std::cout << "player to play\n";
      std::string input;
      std::getline(std::cin, input);

      if (input.starts_with("save")) {
	save_positions(trim_space(input.substr(4)), &board, 1, false);
	break;
      } else if (input.starts_with("undo")) {
	// Next part should be a number, number moves to undo.
	
	for (size_t i {0}; i < std::stoi(input.substr(4)) && board.moves_played > 0; i++) {
	  std::cout << "Undoing move " << board.moves_played_vector[board.moves_played - 1] << '\n';
	  board.undo_move();
	}
	continue;
      } else {
	move.subboard = std::atoi(input.data());
	move.square = std::atoi(input.data() + 1);
      }
    } else {
      std::cout << "bot to play\n";
      if (board.count_total_empty_squares() <= full_search_threshold) {
	std::cout << "trying full search\n";
	move = bot->pick_move_full(board);
      } else {
	move = bot->pick_move(board);
      }
    }

    std::cout << "playing move " << move << '\n';
    if (!board.play_move(move)) {
      std::cout << "Illegal move\n";
      if (board.next_player() != player) {
	// if it was the bot, don't try again.
	save_positions("bot-fail", &board, 1, false);
	break;
      } else {
	// wasn't bot. try again.
	std::cout << "try again\n";
	i--;
      }
    }
  }

  std::cout << board << '\n';
}

template <size_t max_depth, Heuristic eval>
void play_game_with_player (Negamax<max_depth, eval> *bot, const size_t full_search_threshold, const Player player, Board &&board = Board {}) {
  play_game_with_player(bot, full_search_threshold, player, board);
}

int main () {
  try {
    const bool overwrite {false};
    Board::pre_generate_legal_moves(overwrite);
    std::mt19937 rng {};

    //TODO SOMEWHERE ELSE
    // make a function to check if the current position will just result in a draw

    std::vector<Board> random_boards {};
    constexpr size_t num_boards {100'000};
    std::string positions_filename {"positions-" + std::to_string(num_boards)};
    std::cout << positions_filename << '\n';
    if (std::filesystem::exists(positions_filename)) {
      std::cout << "Loading " << positions_filename << '\n';
      random_boards = load_positions(positions_filename, num_boards);
    } else {
      std::cout << "creating " << positions_filename << '\n';
      random_boards.resize(num_boards);
      for (Board &b : random_boards) {
	b = random_position(rng, false);
      }
      const bool append_positions {false};
      save_positions(positions_filename, random_boards.data(), random_boards.size(), append_positions);
    }
    std::cout << "done\n";

    /*
    // For debugging a specific position
    Board board {};
    Move moves[] {
      {0, 0}, {0, 1}, {1, 0}, {0, 2}, {2, 0}, {0, 3}, {3, 0}, {0, 4}, {4, 4}, {4, 0}, {0, 8}, {8, 0}, {0, 7}, {7, 1}, {1, 1}, {1, 2}, {2, 2}, {2, 1}, {1, 4}, {4, 2}, {2, 4}, {4, 1}, {1, 7}, {7, 0}
    };
    for (auto m : moves) {
      if (!board.play_move(m)) {
	std::cout << m << '\n';
	std::cout << "problem\n";
	break;
      }
    }
    std::cout << (int)board.moves_played << '\n';
    std::cout << board << '\n';
    std::cout << "undoing move " << board.moves_played_vector[board.moves_played - 1] << '\n';
    board.undo_move();
    std::cout << (int)board.moves_played << '\n';
    std::cout << board << '\n';
    */

    /*
    Negamax<1, heur3> bot_n {"negamax", Eval_Params {}, rng()};
    play_game_with_player(&bot_n, 22, Player::X);
    //play_game(&bot_n, &bot_n, rng());
    */

    std::vector<Bot_ptr> bots;
    
    bots.push_back(std::make_unique<Negamax<5, heur1>>("5h1", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<5, heur2>>("5h2", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<5, heur3>>("5h3", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<5, heur4>>("5h4", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<6, heur1>>("6h1", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<6, heur2>>("6h2", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<6, heur3>>("6h3", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<6, heur4>>("6h4", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<7, heur1>>("7h1", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<7, heur2>>("7h2", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<7, heur3>>("7h3", Eval_Params {}, rng()));
    bots.push_back(std::make_unique<Negamax<7, heur4>>("7h4", Eval_Params {}, rng()));

    //    std::cout << simulate(bots, 10) << '\n';

    
    // Benchmarking speed of alternate heuristics.
    std::cout << benchmark_heuristic(heur3, "h3", random_boards) << '\n';
    std::cout << benchmark_heuristic(heur4, "h4", random_boards) << '\n';
    auto a = Negamax<3, heur3> {"3h3", Eval_Params {}, rng()};
    auto b = Negamax<3, heur4> {"3h4", Eval_Params {}, rng()};
    auto c = Negamax<3, heur5> {"3h5", Eval_Params {}, rng()};
    auto d = Negamax<5, heur3> {"5h3", Eval_Params {}, rng()};
    std::cout << benchmark_bot_move_generation(a, random_boards) << '\n';
    std::cout << benchmark_bot_move_generation(b, random_boards) << '\n';
    std::cout << benchmark_bot_move_generation(c, random_boards) << '\n';
    std::cout << benchmark_bot_move_generation(d, random_boards) << '\n';


    // Testing how far a full search can go.
    Negamax<7, heur4> full {"7h4", Eval_Params {}, rng()};
    test_full_search(rng, full);
    std::cout << benchmark_bot_move_generation(full, random_boards) << '\n';


    //    evolve_eval_params_for_bot<Negamax<5, heur3>>(100, 12);

  } catch (const std::exception &e) {
    std::cerr << e.what();

    return EXIT_FAILURE;
  }

  return 0;
}
