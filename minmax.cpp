#include <iostream>  // Include the standard input/output library for console I/O
#include <vector>    // Include the vector container from the STL
#include <omp.h>     // Include OpenMP library for parallel processing support

using namespace std; // Use the entire std namespace

int main() {
    // Sample data initialization
    // Create a vector of integers with some sample values
    vector<int> data = {5, 2, 9, 1, 7, 6, 4, 3, 8};
    
    // Initialize reduction variables:
    int sum = 0;             // Will accumulate the sum of all elements
    int min_val = data[0];   // Initialize min with first element
    int max_val = data[0];   // Initialize max with first element
    
    // Parallel reduction using OpenMP:
    // This pragma directive tells the compiler to parallelize the following for loop
    // It specifies three different reduction operations:
    // 1. sum: values will be added together (+ operation)
    // 2. min_val: will track the minimum value (min operation)
    // 3. max_val: will track the maximum value (max operation)
    #pragma omp parallel for reduction(+:sum) reduction(min:min_val) reduction(max:max_val)
    for (size_t i = 0; i < data.size(); i++) {
        // This loop body will be executed in parallel by multiple threads
        sum += data[i];          // Each thread accumulates a partial sum
        if (data[i] < min_val)  // Each thread tracks its local minimum
            min_val = data[i];
        if (data[i] > max_val)  // Each thread tracks its local maximum
            max_val = data[i];
    }
    // After the loop, OpenMP automatically combines:
    // - All partial sums into the final sum
    // - All local minima into the global minimum
    // - All local maxima into the global maximum
    
    // Calculate average by dividing the total sum by number of elements
    // static_cast<double> ensures floating-point division
    double avg = static_cast<double>(sum) / data.size();
    
    // Output results to the console
    cout << "Sum: " << sum << "\n"           // Print sum
         << "Average: " << avg << "\n"       // Print average
         << "Minimum: " << min_val << "\n"   // Print minimum value
         << "Maximum: " << max_val << endl;  // Print maximum value
    
    return 0;  // Exit program with success status
}
