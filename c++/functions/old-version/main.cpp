#include <iostream>
#include <memory>

#include "function.hpp"
#include "token.hpp"

typedef std::shared_ptr<Function> function_pointer;

// Converts a linked list of 3 elements to a tree.
// Assumes that there are at least 3 elements in the linked list.
// Call by having (3) (+) (5)
// (3) = linked_operation_to_tree(3);
// (3) becomes a plus of 3 and 5.
// (3 + 5)
function_pointer linked_operation_to_tree (const function_pointer &beginning) {
  auto root = beginning->get_next();
  auto a = beginning;
  auto b = beginning->node_at(2);
  root->set_a(a);
  root->set_b(b);
  root->set_next(b->get_next());
  return root;
}

function_pointer generate_tree (function_pointer beginning, const function_pointer &end) {
  for (auto p = beginning; p != end; p = p->get_next()) {
    std::cout << *p << " ";
  } std::cout << "\n";
  
  if (beginning->get_next() == end) return beginning;
  
  bool is_simple {true};
  size_t i {0};
  for (auto p = beginning; p != end; p = p->get_next(), i++) {
    if (i % 2 == 0) {
      // Should be an operand.
      if (!is_operand(p)) {
	is_simple = false;
	break;
      }
    } else {
      // Should be an operator.
      if (!is_operator(p)) {
	is_simple = false;
	break;
      }
    }
  }
  
  if (i % 2 == 0) is_simple = false;
  std::cout << "is simple: " << is_simple << "\n";
  if (is_simple) {
    auto root = beginning;
    // Look first for multiplication and division.
    // For each node, if there are 2 more nodes after it:
    //   If the node after it is multiplication or division:
    //     If it is the beginning:
    //       node = tree_triplet(node)
    //       move the beginning to the node (beginning = node)
    //     Otherwise if it is not the beginning:
    //       node = tree_triplet(node)
    //   If the next node is not multiplication or division:
    //     skip forwards 2 nodes.
    
    
    // 3 + x * 5 + 1
    // 3 + (x * 5) + 1
    // + is not updated to point to *, so instead it becomes
    // 3 + x (x * 5) + 1
    for (; root->node_at_exists(2);) {
      if (root->get_next()->get_type() == Function_Type::MULTIPLICATION ||
	  root->get_next()->get_type() == Function_Type::DIVISION) {
	if (beginning == root) {
	  root = linked_operation_to_tree(root);
	  beginning = root;
 	} else {
	  root = linked_operation_to_tree(root);
	}
	std::cout << "root symbol: " << root->get_symbol() << " | root expression: " << *root << "\n";
      } else {
	root = root->node_at(2);
      }
    }
    
    // Print out the linked list.
    for (auto p = beginning; p != end; p = p->get_next()) {
      std::cout << *p << "\n";
    } std::cout << "\n";
    
    root = beginning;
    
    return root;
  }
  
  return nullptr;
}

function_pointer build_function (const std::vector<Token> &tokens) {
  function_pointer function {token_to_function(tokens[0])};
  
  // Converts the vector of tokens to a linked list of functions.
  for (size_t i {1}; i < tokens.size(); i++) {
    function->node_at(i) = token_to_function(tokens[i]);
  }
  
  function = generate_tree(function, nullptr);
  
  return function;
}

int main () {
  // Current list of possible operations:
  const std::vector<std::string> operations {"+", "*", "-", "/", "^", "(", ")"};
  
  //                     0123456789
  std::string function {"9 + 5/7 * 3 - x * 5 + 1"};
  std::vector<Token> tokenized_function {tokenize(function, operations)};
  for (const auto &token : tokenized_function) {
    std::cout << token << "\n";
  }
  std::cout << "\n";
  
  auto g = build_function(tokenized_function);
  std::cout << "Formula of g: " << *g << "\n";
  
  /*auto f = make_function<Exponent>(make_function<Addition>(make_constant(2), make_constant(5.1)), make_variable("X"));
    
    std::cout << f->evaluate(3) << "\n";
    
    // Print the tree of f.
    std::cout << *f << "\n";*/
  
  return 0;
}
