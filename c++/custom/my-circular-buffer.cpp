#include <iostream>

template <class T>
class Circular_Buffer {
private:
    T *buffer {};
    int _buffer_size {};
    
    // Points to the first element.
    T *start {};
    // Points to after the last element.
    T *end {};
    
    // The number of elements that are being stored in the buffer.
    int _elements_used {};
public:
    Circular_Buffer (int size) {
        _buffer_size = size;
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
        if (_elements_used < _buffer_size) {
            _elements_used++;
            
            *end = value;
            
            end++;
            
            // If end points to after the end of the array, then wrap around to the beginning.
            if (end >= buffer + _buffer_size) {
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
        if (_elements_used + size <= _buffer_size) {
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
        while (elements_added < size && _elements_used < _buffer_size) {
            add_element(values[elements_added]);
            elements_added++;
        }
    }
    
    // Removes an element from the beginning.
    void remove_first_element () {
        // Only remove an element if the number of elements is greater then 0.
        if (_elements_used > 0) {
            _elements_used--;
            
            start++;
            
            // If start points to after the end of the array, then wrap around to the beginning.
            if (start >= buffer + _buffer_size) {
                start = buffer;
            }
        } else {
            // If there are no elements to remove, throw an exception.
            throw std::out_of_range("Buffer is empty. Cannot remove first element");
        }
    }
    
    T& read_first_element () {
        Only read if there are elements stored.
        if (_elements_used > 0) {
            return *start;
        }
        
        throw std::out_of_range("Buffer is empty. Cannot read first element.");
    }
    
    T& destructive_read_first_element () {
        T &temp_value = read_first_element();
        remove_first_element();
        return temp_value;
    }
    
    int buffer_size () {
        return _buffer_size;
    }
    
    int elements_used () {
        return _elements_used;
    }
};

int main () {
    return 0
}
