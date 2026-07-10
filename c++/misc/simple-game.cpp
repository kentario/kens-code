#include <iostream>
#include <string>
#include <initializer_list>
#include <vector>
#include <array>
#include <span>
#include <cstdint>
#include <cfloat>
#include <memory>
#include <fstream>
#include <cassert>
#include <stdexcept>

#include <sstream>
#include <iomanip>

#define DEBUG false
#define debug(x) if constexpr (DEBUG) { x; }

// Forward declaration.
struct Game {
  std::vector<uint8_t> moves {};
  Game (std::initializer_list<uint8_t> &&moves);
  void play (const bool move);
  Game play_value (const bool move) const;
  void play_moves (const std::span<uint8_t> moves);
  void undo_move ();
  uint8_t seq_to_index (const std::vector<uint8_t> &seq) const;
  int moves_contain_seq_starting_from (const std::vector<uint8_t> &seq, const size_t s) const;

  template <size_t SEQ_SIZE, size_t NUM_SEQ>
  bool check_win () const;
};

std::ostream& operator<< (std::ostream &os, const Game &game) {
  for (const auto m : game.moves) {
    os << static_cast<int>(m);
  }
  return os;
}
std::ostream& operator<< (std::ostream &os, const std::vector<uint8_t> &seq) {
  for (const auto m : seq) {
    os << static_cast<int>(m);
  }
  return os;
}
template <size_t S>
std::ostream& operator<< (std::ostream &os, const std::array<bool, S> &arr) {
  for (const auto m : arr) {
    os << static_cast<int>(m);
  }
  return os;
}

// Actual contents.
Game::Game (std::initializer_list<uint8_t> &&moves) : moves {moves} {}

void Game::play (const bool move) {
  moves.push_back(move);
}
Game Game::play_value (const bool move) const {
  Game result {*this};
  result.play(move);
  return result;
}
void Game::play_moves (const std::span<uint8_t> moves) {
  for (const auto m : moves) {
    play(m);
  }
}
void Game::undo_move () {
  moves.pop_back();
}

uint8_t Game::seq_to_index (const std::vector<uint8_t> &seq) const {
  uint8_t res {0};
  for (const uint8_t m : seq) {
    res <<= 1;
    res |= static_cast<bool>(m);
  }

  return res;
}

// Starting from (including) index s, check if the moves vector contains any instances of seq.
// If so, return the starting index of the first occurence of seq.
// Otherwise, return -1.
int Game::moves_contain_seq_starting_from (const std::vector<uint8_t> &seq, const size_t s) const {
  if (s + seq.size() > moves.size()) {
    debug(std::cout << "not enough room for seq\n")
    return -1;
  }
  debug(
	std::cout << "from moves_contain_seq_starting_from\n";
  std::cout << "    starting at " << s << '\n';
  std::cout << "    " << std::vector<uint8_t>(moves.begin() + s, moves.end()) << '\n';
  )
  // For each possible starting point.
  for (size_t i {s}; i <= moves.size() - seq.size(); i++) {
    // Check if seq exists from that starting point.
    bool match {true};
    for (size_t offset {0}; offset < seq.size(); offset++) {
      if (seq[offset] != moves[i + offset]) {
	match = false;
	break;
      }
    }
    if (match) {
      debug(std::cout << "found at " << i << '\n')
      return i;
    }
  }
  debug(std::cout << "none found\n")
  return -1;
}

template <size_t SEQ_SIZE, size_t NUM_SEQ>
bool Game::check_win () const {
  if (moves.size() < SEQ_SIZE * NUM_SEQ) {
    debug(std::cout << "moves not big enough\n")
    return false;
  }

  // List of all SEQ_SIZE sequences that have been found not to be a win.
  // There will be 2^SEQ_SIZE possible sequences.
  // 2^int = 1 << int
  std::array<bool, (1 << SEQ_SIZE)> already_rejected {};

  // The last starting point to check is the last index where including that index, there are SEQ_SIZE * NUM_SEQ moves remaining.
  for (size_t i {0}; i <= moves.size() - SEQ_SIZE * NUM_SEQ; i++) {
    // get subvector
    std::vector<uint8_t> seq(moves.begin() + i, moves.begin() + i + SEQ_SIZE);
    debug(
	  std::cout << "\nNew search of seq " << seq << " starting at " << i << '\n';
    std::cout << "current already_rejected:       " << already_rejected << '\n';
    )
      
    auto &is_not_win = already_rejected[seq_to_index(seq)];
    // Search for occurences of the subvector.
    // If this is a know wrong sequence, then skip it.
    if (is_not_win) {
      debug(std::cout << "This seq is known to not be a win, skipping\n")
      continue;
    }
    // Otherwise, search for it.
    // Sequences cannot overlap.
    // must find a total of NUM_SEQ sequences, with 1 already found.
    size_t last_seq_start {i};
    for (size_t num_found {1}; num_found < NUM_SEQ; num_found++) {
      debug(
	    std::cout << "found " << num_found << " so far\n";
      std::cout << "need to find a total of " << NUM_SEQ << '\n';
      )
      const auto index_of_next_seq = moves_contain_seq_starting_from(seq, last_seq_start + SEQ_SIZE);
      if (index_of_next_seq != -1) {
	last_seq_start = index_of_next_seq;
      } else {
	// A total of 4 sequences were not found.
	// Therefore this sequence is not valid.
	is_not_win = true;
	// No need to keep searching.
	break;
      }
    }
    if (is_not_win) continue;

    debug(
	  std::cout << "found win, already_rejected: " << already_rejected << '\n';
    std::cout << "The seq is " << seq << '\n';
    )
    // If at the end of the loop, and the sequence is known to not not be a win, then it must be a win.
    return true;
  }

  debug(std::cout << "\nno win found, already_rejected: " << already_rejected << '\n')
  return false;
}

struct Node {
  // 0 => draw, -1 => player 1 win, 1 => player 2 win
  Game game {};
  int value {};
  std::array<std::unique_ptr<Node>, 2> children {};
};

std::unique_ptr<Node> minimax (const Game game, int depth = -1) {
  auto node = std::make_unique<Node>();
  node->game = game;

  if (depth == 0) return node;
  // Player 1 win.
  if (game.check_win<5, 2>()) {
    node->value = -1;
    // No need to populate the children, as this is terminal.
    return node;
  }
  // Player 2 win.
  if (game.check_win<3, 4>()) {
    node->value = 1;
    return node;
  }

  // Player 1 is about to play.
  const bool to_play_is_min {game.moves.size() % 2 == 0};
  if (to_play_is_min) {
    node->value = 2;
    for (const bool move : {false, true}) {
      node->children[move] = minimax(game.play_value(move), depth - 1);
      if (node->children[move]->value < node->value) {
	// This move is better than others found.
	node->value = node->children[move]->value;
      }
    }
  } else {
    // Player 2 is about to play
    node->value = -2;
    for (const bool move : {false, true}) {
      node->children[move] = minimax(game.play_value(move), depth - 1);
      if (node->children[move]->value > node->value) {
	node->value = node->children[move]->value;
      }
    }
  }

  return node;
}

// bit 1: value      0 = player 1 win (-1), 1 = player 2 win (+1)
// bit 0: children   0 = leaf, 1 = has 2 children.
uint8_t encode_node (const Node &node) {
  // Either both children exist or neither exist.
  const bool child {static_cast<bool>(node.children[0])};
  assert(child == static_cast<bool>(node.children[1]));

  // Should be no draws.
  assert(node.value == -1 || node.value == 1);

  // Nodes are only valued -1 or 1. 0 means -1. 1 means 1.
  return static_cast<uint8_t>((node.value == 1) << 1) |
	 static_cast<uint8_t>(child);
}

class BitWriter {
private:
  std::ostream& out;
  uint8_t buffer {};
  uint8_t bits_used {};

  void flush_byte() {
    out.put(static_cast<char>(buffer));
    buffer = 0;
    bits_used = 0;
  }

public:
  explicit BitWriter(std::ostream& out)
  : out(out) {}

  void write_2_bits(uint8_t value) {
    assert(value < 4);

    buffer |= static_cast<uint8_t>(value << (6 - bits_used));
    bits_used += 2;

    if (bits_used == 8) {
      flush_byte();
    }
  }

  void finish() {
    if (bits_used != 0) {
      // Remaining unused bits are zero padding.
      // Reader must stop after decoding the root subtree, not EOF.
      flush_byte();
    }
  }
};

uint64_t count_nodes (const Node &node) {
  if (!node.children[0]) return 1;

  return 1 + count_nodes(*node.children[0]) + count_nodes(*node.children[1]);
}

void write_tree_preorder(BitWriter& writer, const Node& node) {
  writer.write_2_bits(encode_node(node));

  if (node.children[0]) {
    write_tree_preorder(writer, *node.children[0]);
    write_tree_preorder(writer, *node.children[1]);
  }
}

std::string encode_tree_to_bytes(const Node& root) {
  std::ostringstream out(std::ios::binary);

  BitWriter writer(out);
  write_tree_preorder(writer, root);
  writer.finish();

  return out.str();
}

std::string base64_encode(const std::string& input) {
  static constexpr char alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);

  for (size_t i = 0; i < input.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(input[i]);

    const uint32_t b =
      (i + 1 < input.size())
      ? static_cast<unsigned char>(input[i + 1])
      : 0;

    const uint32_t c =
      (i + 2 < input.size())
      ? static_cast<unsigned char>(input[i + 2])
      : 0;

    const uint32_t combined = (a << 16) | (b << 8) | c;

    output.push_back(alphabet[(combined >> 18) & 0x3F]);
    output.push_back(alphabet[(combined >> 12) & 0x3F]);

    output.push_back(
		     i + 1 < input.size()
		     ? alphabet[(combined >> 6) & 0x3F]
		     : '='
		     );

    output.push_back(
		     i + 2 < input.size()
		     ? alphabet[combined & 0x3F]
		     : '='
		     );
  }

  return output;
}

void write_tree_html(const Node& root, const std::string& filename) {
  const std::string binary_tree = encode_tree_to_bytes(root);
  const std::string base64_tree = base64_encode(binary_tree);

  std::ofstream out(filename);

  if (!out) {
    throw std::runtime_error("Could not open HTML output file");
  }

  out << R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Game Tree Viewer</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      margin: 24px;
    }

    .node {
      border-left: 1px solid #aaa;
      margin: 8px 0 8px 18px;
      padding-left: 12px;
    }

    .header {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .p1 {
      color: #a00000;
    }

    .p2 {
      color: #006000;
    }

    code {
      min-width: 100px;
    }

    button {
      cursor: pointer;
    }

    .children {
      margin-top: 6px;
    }
  </style>
</head>
<body>
  <h1>Game Tree Viewer</h1>
  <p id="status">Loading embedded tree...</p>
  <div id="tree"></div>

  <script>
    const treeBase64 = ")HTML";

  out << base64_tree;

  out << R"HTML(";

    function base64ToBytes(base64) {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);

      for (let i = 0; i < binary.length; ++i) {
        bytes[i] = binary.charCodeAt(i);
      }

      return bytes;
    }

    const bytes = base64ToBytes(treeBase64);

    /*
      Each byte contains four nodes:

      node 0: bits 7-6
      node 1: bits 5-4
      node 2: bits 3-2
      node 3: bits 1-0

      Code bit 1: value
        0 = Player 1 win (-1)
        1 = Player 2 win (+1)

      Code bit 0: child flag
        0 = leaf
        1 = exactly two children
    */
    function nodeCode(nodeIndex) {
      const byteIndex = Math.floor(nodeIndex / 4);
      const positionInByte = nodeIndex % 4;
      const shift = 6 - 2 * positionInByte;

      return (bytes[byteIndex] >> shift) & 0b11;
    }

    function valueOf(code) {
      return (code & 0b10) ? 1 : -1;
    }

    function hasChildren(code) {
      return (code & 0b01) !== 0;
    }

    /*
      The packed tree is preorder:

        node
        left subtree
        right subtree

      This returns the preorder index immediately after the subtree
      beginning at startIndex.
    */
    function skipSubtree(startIndex) {
      const code = nodeCode(startIndex);
      let nextIndex = startIndex + 1;

      if (hasChildren(code)) {
        nextIndex = skipSubtree(nextIndex);
        nextIndex = skipSubtree(nextIndex);
      }

      return nextIndex;
    }

    function makeNode(index, path) {
      const code = nodeCode(index);

      return {
        index: index,
        path: path,
        value: valueOf(code),
        internal: hasChildren(code),
        children: null
      };
    }

    function getChildren(node) {
      if (!node.internal) {
        return [];
      }

      if (node.children !== null) {
        return node.children;
      }

      const falseChildIndex = node.index + 1;
      const trueChildIndex = skipSubtree(falseChildIndex);

      node.children = [
        makeNode(falseChildIndex, node.path + "0"),
        makeNode(trueChildIndex, node.path + "1")
      ];

      return node.children;
    }

    function labelForValue(value) {
      return value === -1 ? "Player 1 wins" : "Player 2 wins";
    }

    function renderNode(node) {
      const element = document.createElement("div");
      element.className = "node";

      const header = document.createElement("div");
      header.className = "header";

      const path = document.createElement("code");
      path.textContent =
        node.path === "" ? "(root)" : node.path;

      const value = document.createElement("strong");
      value.className = node.value === -1 ? "p1" : "p2";
      value.textContent =
        labelForValue(node.value) + " (" + node.value + ")";

      header.append(path, value);
      element.appendChild(header);

      if (!node.internal) {
        return element;
      }

      const button = document.createElement("button");
      button.textContent = "Expand";
      header.appendChild(button);

      const childrenElement = document.createElement("div");
      childrenElement.className = "children";
      childrenElement.hidden = true;
      element.appendChild(childrenElement);

      let rendered = false;

      button.addEventListener("click", () => {
        if (!rendered) {
          const children = getChildren(node);

          for (let move = 0; move < 2; ++move) {
            const branch = document.createElement("div");
            branch.textContent = "Move " + move + ": ";
            branch.appendChild(renderNode(children[move]));
            childrenElement.appendChild(branch);
          }

          rendered = true;
        }

        childrenElement.hidden = !childrenElement.hidden;
        button.textContent =
          childrenElement.hidden ? "Expand" : "Collapse";
      });

      return element;
    }

    const totalNodes = skipSubtree(0);

    document.getElementById("status").textContent =
      "Loaded " + totalNodes + " nodes from this HTML file.";

    const root = makeNode(0, "");
    document.getElementById("tree").appendChild(renderNode(root));
  </script>
</body>
</html>
	  )HTML";

  if (!out) {
    throw std::runtime_error("Error while writing HTML file");
  }
}

int main () {
  // Current game type
  // 2 instances of length 5 = player 1 wins.
  // 4 instances of length 3 = player 2 wins.

  std::cout << "Starting exploration\n";
  auto small_tree = minimax(Game{});
  std::cout << "Starting writing to file\n";
  // All functions encode node and lower are courtesy of chat gpt.
  write_tree_html(*small_tree, "full-tree.html");
  std::cout << "Finished writing to file\n";

  return 0;
}
