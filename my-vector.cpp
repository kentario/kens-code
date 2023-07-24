#include <iostream>

template <class T>
class Vector {
private:
    T *vector {}; // Pointer to the elements stored in the vector.
    int size_ {}; // The size of the vector.
    int capacity_ {}; // The amount of allocated memory in number of elements of the vector.
public:
    // Constructor with no inputs.
    Vector () {}
    
    // Constructor with one input.
    Vector (const int size) {
        size_ = size;
        capacity_ = size;
        vector = new T[capacity_];
    }
    
    // Constructor with two inputs.
    Vector (const int size, const T &value) {
        size_ = size;
        capacity_ = size;
        vector = new T[capacity_];
        
        for (int i {0}; i < size; i++) {
            vector[i] = value;
        }
    }
    
    Vector (const std::initializer_list<T> &list) {
        size_ = list.size();
        capacity_ = list.size();
        vector = new T[capacity_];
        
        int i {0};
        for (const T &element : list) {
            vector[i] = element;
            i++;
        }
    }
    
    // Destructor.
    ~Vector () {
        delete[] vector;
    }
    
    // Resize the vector to a new size.
    void resize (const int &size) {
        // If the new size is greater than the capacity, then allocate some memory.
        if (size > capacity_) {
            try {
                const T *temp_vector = vector;
                // If the memory for the vector had to be allocated, then the size is greater then the old capacity, so the capacity value needs to be updated.
                capacity_ = size;
                vector = new T[capacity_];
                
                // Copy everything from the old vector.
                for (int i {0}; i < size_; i++) {
                    vector[i] = temp_vector[i];
                }
                
                // Delete the old vector.
                delete[] temp_vector;
            }
            catch (const std::bad_alloc& e) {
                std::cout << "Memory Allocation failed: " << e.what() << "\n";
                exit(1);
            }
        }
        // If the size of the vector grows, then fill in all the new space with nothing.
        if (size > size_) {
            for (int i {size_}; i < size; i++) {
                // For each new element, fill it with the default value of T.
                vector[i] = T{};
            }
        }
        // The value of size_ needs to be updated regardless of whether any memory was allocated.
        size_ = size;
    }
    
    bool operator== (const Vector<T> &other_vector) {
        // If they have different sizes, they are not equal.
        if (size_ != other_vector.size_) {
            return false;
        }
        // If any of their elements are not equal, they are not equal.
        for (int i {0}; i < size_; i++) {
            if (vector[i] != other_vector.vector[i]) {
                return false;
            }
        }
        // Otherwise they are equal.
        return true;
    }

    bool operator== (const std::initializer_list<T> &list) {
        // If they have different sizes, they are not equal.
        if (size_ != list.size()) {
            return false;
        }
        // If any of their elements are not equal, they are not equal.
        int i {0};
        for (const T &element : list) {
            if (vector[i] != element) return false;
            i++;
        }
        
        // Otherwise they are equal.
        return true;
    }
    
    Vector<T>& operator= (const Vector<T> &other_vector) {
        if (*this == other_vector) {
            return *this;
        }
        if (other_vector.size_ > capacity_) {
            try {
                // Delete the vector.
                delete[] vector;
                // If the memory for the vector had to be allocated, then the size is greater then the old capacity, so the capacity value needs to be updated.
                capacity_ = other_vector.size_;
                // Resize the vector to the new vectors size.
                vector = new T[capacity_];
            }
            catch (const std::bad_alloc& e) {
                std::cout << "Memory Allocation failed: " << e.what() << "\n";
                exit(1);
            }
        }
        // Update the size value regardless of whether memory needs to be allocated.
        size_ = other_vector.size_;
        // Copy the data from the other vector regardless of whehter memory needs to be allocated.
        for (int i {0}; i < size_; i++) {
            vector[i] = other_vector.vector[i];
        }
        // "this" is a pointer to the object that used "this".
        return *this;
    }

    Vector<T>& operator= (const std::initializer_list<T> &list) {
        if (*this == list) {
            return *this;
        }
        if (list.size() > capacity_) {
            try {
                // Delete the vector.
                delete[] vector;
                // If the memory for the vector had to be allocated, then the size is greater then the old capacity, so the capacity value needs to be updated.
                capacity_ = list.size();
                // Resize the vector to the new vectors size.
                vector = new T[capacity_];
            }
            catch (const std::bad_alloc& e) {
                std::cout << "Memory Allocation failed: " << e.what() << "\n";
                exit(1);
            }
        }
        // Update the size value regardless of whether memory needs to be allocated.
        size_ = list.size();
        // Copy the data from the other vector regardless of whether memory needs to be allocated.
        int i {0};
        for (const T &element : list) {
            vector[i] = element;
            i++;
        }
        
        // "this" is a pointer to the object that used "this".
        return *this;
    }
    
    // Accessing parts of the vector.
    T& operator[] (const int index) {
        if (index < 0 || index >= size_) {
            //throw std::out_of_range("Index out of range");
        }
        return vector[index];
    }
    
    // Access the first element.
    T& front () {
        return vector[0];
    }
    
    // Access the last element.
    T& back () {
        return vector[size_ - 1];
    }
    
    // Returns whether the vector is empty or not.
    bool empty () {
        return (size_ > 0) ? false : true;
    }
    
    // Add an element to the end.
    void push_back (const T &value) {
        // Increase the size of the vector.
        resize(size_ + 1);
        // Assign the last element to be value.
        back() = value;
    }
    
    // Remove the last element.
    void pop_back () {
        // Only shrink the vector when it is not empty.
        if (!empty()) size_--;
    }
    
    // Returns a pointer to the array used to store the the vectors elements.
    T* data () {
        return vector;
    }
    
    int size () {
        return size_;
    }
    
    int capacity () {
        return capacity_;
    }
};

int main () {
  return 0;
}
