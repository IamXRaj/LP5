#include <iostream>  // Include the standard input/output library for console I/O
#include <vector>    // Include the vector container from the STL
#include <omp.h>     // Include OpenMP library for parallel processing support

int main() {
    // Sample data initialization
    // Create a vector of integers with some sample values
    std::vector<int> data = {5, 2, 9, 1, 7, 6, 4, 3, 8};
    
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
    std::cout << "Sum: " << sum << "\n"           // Print sum
              << "Average: " << avg << "\n"       // Print average
              << "Minimum: " << min_val << "\n"   // Print minimum value
              << "Maximum: " << max_val << std::endl;  // Print maximum value
    
    return 0;  // Exit program with success status
}



/*
### **Key Terms Related to the Code**  

1. **OpenMP (`#pragma omp`)**  
   - API for **parallel programming** in C/C++/Fortran.  
   - Uses compiler directives to enable multi-threading.  

2. **Reduction Operation**  
   - Combines partial results from multiple threads into a single final result.  
   - Supported operations: `+`, `min`, `max`, `*`, `&`, `|`, etc.  

3. **Parallel For Loop (`#pragma omp parallel for`)**  
   - Splits loop iterations across threads for concurrent execution.  

4. **Reduction Variables (`reduction` clause)**  
   - **`sum`**: Each thread computes a partial sum, combined at the end.  
   - **`min_val`**: Each thread finds its local minimum, final result is global min.  
   - **`max_val`**: Each thread finds its local maximum, final result is global max.  

5. **Race Condition Prevention**  
   - **`reduction`** avoids data races by handling variable updates safely.  
   - No need for explicit locks (e.g., `atomic` or `critical`).  

6. **Static Cast (`static_cast<double>`)**  
   - Ensures **floating-point division** for accurate average calculation.  

7. **Performance Benefits**  
   - Parallel reduction speeds up computations for large datasets.  
   - Threads work on different parts of the data simultaneously.  

8. **Thread Safety**  
   - OpenMP automatically handles synchronization for reduction variables.  

9. **Data Distribution**  
   - Loop iterations are divided **evenly** among threads (default scheduling).  

10. **Output Synchronization**  
    - `std::cout` is thread-safe but may interleave output without synchronization.  

### **Why This Code Matters**  
- Demonstrates **efficient parallel reduction** (sum, min, max, avg).  
- Shows how OpenMP simplifies parallel programming with **automatic thread management**.  
- Useful for statistical computations on large datasets.  

### **Execution Flow**  
1. **Initialize** data and reduction variables.  
2. **Parallel loop**: Threads compute partial sums, mins, and maxes.  
3. **Combine results**: OpenMP merges partial results into final values.  
4. **Calculate average** (sequential step).  
5. **Print results** (sum, avg, min, max).  

This example is a **foundational pattern** in parallel computing, often used in numerical analysis and data processing.

*/