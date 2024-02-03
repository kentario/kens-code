#include <memory>
#include <cstdarg>

#pragma once

template <typename T>
struct Node : public std::enable_shared_from_this<Node<T>> {
  T data {};
  std::shared_ptr<Node<T>> prev {};
  std::shared_ptr<Node<T>> next {};

  explicit Node (const T &data) : data {data} {}
  Node (const T &data, const std::shared_ptr<Node<T>> &prev, const std::shared_ptr<Node<T>> &next = nullptr) : data {data}, prev {prev}, next {next} {}

  void insert_node (std::shared_ptr<Node<T>> before, std::shared_ptr<Node<T>> after);
  void replace_node (std::shared_ptr<Node<T>> first, std::shared_ptr<Node<T>> last);

  std::shared_ptr<Node<T>>& node_at (int index);

  bool node_at_exists (int index);
};

template <typename T>
void Node<T>::insert_node (std::shared_ptr<Node<T>> before, std::shared_ptr<Node<T>> after) {
  prev = before;
  next = after;
  if (before) {
    before->next = this->shared_from_this();
  }
  if (after) {
    after->prev = this->shared_from_this();
  }
}

template <typename T>
void Node<T>::replace_node (std::shared_ptr<Node<T>> first, std::shared_ptr<Node<T>> last) {

  if (first) {
    prev = first->prev;
    if (first->prev) {
      first->prev->next = this->shared_from_this();
    }
  }
  if (last) {
    next = last->next;
    if (last->next) {
      last->next->prev = this->shared_from_this();
    }
  }
}

template <typename T>
std::shared_ptr<Node<T>>& Node<T>::node_at (int index) {
  if (index > 0) {
    if (index == 1) return next;
    if (next) return next->node_at(index - 1);
  } else {
    if (index == -1) return prev;
    if (prev) return prev->node_at(index + 1);
  }
  
  throw std::out_of_range {"Index out of range"};
}

template <typename T>
bool Node<T>::node_at_exists (int index) {
  if (index == 0) return true;
  if (index > 0) {
    if (index == 1) return next.get();
    if (next) return next->node_at_exists(index - 1);
  } else {
    if (index == -1) return prev.get();
    if (prev) return prev->node_at_exists(index + 1);
  }
    
  return false;
}

// Base case for recursion.
template <typename T>
std::shared_ptr<Node<T>> make_linked_list_helper () {
  return nullptr;
}

template <typename T, typename... Args>
std::shared_ptr<Node<T>> make_linked_list_helper (const T &first, const Args &...args) {
  auto node = std::make_shared<Node<T>>(first);
  if constexpr (sizeof...(args) > 0) {
    node->next = make_linked_list_helper(args...);
    if (node->next) {
      node->next->prev = node;
    }
  }
  
  return node;
}

template <typename T, typename... Args>
std::shared_ptr<Node<T>> make_linked_list(const T &first, const Args &...args) {
    return make_linked_list_helper(first, args...);
}

// Inserts a new node with the given data inbetween before and after. before and after are both still part of the linked list.
// Returns a pointer to the new node.
template <typename T>
std::shared_ptr<Node<T>> insert_node (std::shared_ptr<Node<T>> before, std::shared_ptr<Node<T>> after, const T &new_data) {
  auto new_node = std::make_shared<Node<T>>(new_data, before, after);
  if (before) {
    before->next = new_node;
  }
  if (after) {
    after->prev = new_node;
  }

  return new_node;
}

// Replaces the range between first and last with a new node containing the data. first and last are removed.
// Returns a pointer to the new node.
template <typename T>
std::shared_ptr<Node<T>> replace_node (std::shared_ptr<Node<T>> first, std::shared_ptr<Node<T>> last, const T &new_data) {
  auto new_node = std::make_shared<Node<T>>(new_data);
  if (!first || !last) throw std::invalid_argument {"Unexpected nullptr"};
  
  new_node->prev = first->prev;
  if (first->prev) {
    first->prev->next = new_node;
  }
  new_node->next = last->next;
  if (last->next) {
    last->next->prev = new_node;
  }
  
  return new_node;
}
