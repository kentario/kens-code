#include <iostream>

template <class T>
class Circular_Buffer {
private:
    T *buffer {};
    int buffer_size_ {};
    
    // Points to the first element.
    T *start {};
    // Points to after the last element.
    T *end {};
    
    // The number of elements that are being stored in the buffer.
    int elements_used_ {};
public:
    Circular_Buffer (int size) {
        buffer_size_ = size;
        buffer = new T[size];
        // Both start and end point to the beginning of the buffer.
        // When start and end point to the same location, that means that the buffer is empty.
        start = buffer;
        end = buffer;
    }
    
    ~Circular_Buffer () {
        delete[] buffer;
    }
    
    // Adds an element to the end.
    void add_element (const T &value) {
        // Only add an element if the number of elements is less then the maximum size.
        if (elements_used_ < buffer_size_) {
            elements_used_++;
            
            *end = value;
            
            end++;
            
            // If end points to after the end of the array, then wrap around to the beginning.
            if (end >= buffer + buffer_size_) {
                end = buffer;
            }
        } else {
            // If the buffer is full, then throw an exception.
            throw std::out_of_range("Buffer is full. Cannot add new element");
        }
    }
    
    // Adds an array of elements to the end, if the array doesn't fit, then add nothing.
    void add_array_safe (const T *values, int size) {
        // Only add the elements if they all fit in the buffer.
        if (elements_used_ + size <= buffer_size_) {
            for (int i {0}; i < size; i++) {
                add_element(values[i]);
            }
        } else {
            throw std::out_of_range("Array size exceeds buffer capacity");
        }
    }
    
    // Adds an array of elements to the end, if the array doesn't fit, then adds everything that does fit.
    void add_array_best_effort (const T *values, int size) {
        int elements_added {};
        
        // Keep adding elements while there are elements to add, and there is still space left in the buffer.
        while (elements_added < size && elements_used_ < buffer_size_) {
            add_element(values[elements_added]);
            elements_added++;
        }
    }
    
    // Removes an element from the beginning.
    void remove_first_element () {
        // Only remove an element if the number of elements is greater then 0.
        if (elements_used_ > 0) {
            elements_used_--;
            
            start++;
            
            // If start points to after the end of the array, then wrap around to the beginning.
            if (start >= buffer + buffer_size_) {
                start = buffer;
            }
        } else {
            // If there are no elements to remove, throw an exception.
            throw std::out_of_range("Buffer is empty. Cannot remove first element");
        }
    }
    
    T& read_first_element () {
      // Only read if there are elements stored.
      if (elements_used_ > 0) {
	return *start;
      }
        
      throw std::out_of_range("Buffer is empty. Cannot read first element.");
    }
    
    T& destructive_read_first_element () {
        T &temp_value = read_first_element();
        remove_first_element();
        return temp_value;
    }
    
    int buffer_size () const {
        return buffer_size_;
    }
    
    int elements_used () const {
        return elements_used_;
    }
};

int main (int argc, char *argv[]) {
  return 0;
}
