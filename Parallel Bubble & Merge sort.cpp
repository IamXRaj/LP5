#include <iostream>    // Standard input/output operations
#include <vector>      // Dynamic array container
#include <algorithm>   // For swap function
#include <omp.h>       // OpenMP library for parallel processing

using namespace std; // Use the entire std namespace

// Parallel Bubble Sort using Odd-Even Transposition algorithm
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();       // Get array size
    bool swapped;             // Flag to track if swaps occurred
    
    // Outer loop for each sorting pass
    for (int i = 0; i < n; ++i) {
        swapped = false;      // Reset swap flag for new pass
        
        // Parallel inner loop - compares adjacent elements
        #pragma omp parallel for shared(arr, swapped)
        // Odd-even approach: alternates starting points (0 or 1)
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {      // Compare neighbors
                swap(arr[j], arr[j + 1]); // Swap if out of order
                #pragma omp atomic write      // Thread-safe flag update
                swapped = true;               // Mark swap occurred
            }
        }
        
        // Early termination if no swaps in pass (array is sorted)
        if (!swapped) break;
    }
}





// Sequential Merge helper function for merge sort
void merge(vector<int>& arr, int l, int m, int r) {
    // Create temp array containing elements to merge
    vector<int> temp(arr.begin() + l, arr.begin() + r + 1);
    
    // Initialize pointers:
    int i = 0;              // Left subarray (starts at temp[0])
    int j = m - l + 1;      // Right subarray (starts at temp[m-l+1])
    int k = l;              // Position in original array
    
    // Merge while both subarrays have elements
    while (i <= m - l && j <= r - l) {
        // Select smaller element from either subarray
        arr[k++] = (temp[i] <= temp[j]) ? temp[i++] : temp[j++];
    }
    
    // Copy remaining left subarray elements if any
    while (i <= m - l) arr[k++] = temp[i++];
}

// Parallel Merge Sort using divide-and-conquer
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {  // Base case: more than one element
        int m = l + (r - l) / 2;  // Calculate midpoint
        
        // Parallel recursive sorting:
        #pragma omp parallel sections  // Split into parallel sections
        {
            #pragma omp section        // First thread sorts left half
            parallelMergeSort(arr, l, m);
            
            #pragma omp section        // Second thread sorts right half
            parallelMergeSort(arr, m + 1, r);
        }
        
        merge(arr, l, m, r);  // Merge the sorted halves
    }
}

int main() {
    // Initialize test data
    vector<int> arr = {5, 2, 9, 1, 5, 6};
    vector<int> arr_copy = arr;  // Duplicate for merge sort

    // Parallel Bubble Sort demo
    cout << "Before Bubble Sort: ";
    for (int num : arr) cout << num << " ";  // Print original
    
    parallelBubbleSort(arr);  // Perform parallel bubble sort
    
    cout << "\nAfter Bubble Sort:  ";
    for (int num : arr) cout << num << " ";  // Print sorted

    // Parallel Merge Sort demo
    cout << "\n\nBefore Merge Sort:  ";
    for (int num : arr_copy) cout << num << " ";  // Print original
    
    parallelMergeSort(arr_copy, 0, arr_copy.size() - 1);  // Perform merge sort
    
    cout << "\nAfter Merge Sort:   ";
    for (int num : arr_copy) cout << num << " ";  // Print sorted

    return 0;  // Exit program
}


/*

### **Key Terms Related to the Code:**  

1. **Bubble Sort**  
   - Simple sorting algorithm that repeatedly swaps adjacent elements if they are in the wrong order.  
   - **Time Complexity**: \(O(n^2)\) (worst/average case).  

2. **Odd-Even Transposition Sort**  
   - Parallel version of Bubble Sort where adjacent elements are compared in alternating (odd-even) phases.  
   - Reduces sequential dependencies for better parallelism.  

3. **Merge Sort**  
   - Divide-and-conquer algorithm that splits the array, sorts subarrays, and merges them.  
   - **Time Complexity**: \(O(n \log n)\) (worst/average case).  

4. **Parallel Sorting**  
   - Uses multiple threads to speed up sorting (e.g., OpenMP).  
   - **Bubble Sort Parallelism**: Odd-even phases allow concurrent swaps.  
   - **Merge Sort Parallelism**: Recursive subarray sorting in parallel.  

5. **OpenMP (`#pragma omp` Directives)**  
   - **`parallel for`**: Parallelizes loop iterations across threads.  
   - **`sections`**: Divides work into independent blocks for different threads.  
   - **`atomic`**: Ensures thread-safe updates (e.g., `swapped` flag).  

6. **Race Condition**  
   - When multiple threads modify shared data (e.g., `swapped` flag) without synchronization.  
   - Prevented using **`atomic`** or **`critical`** sections.  

7. **Divide-and-Conquer**  
   - Merge Sort splits the problem into smaller subproblems (recursively).  
   - Parallelism comes from sorting subarrays independently.  

8. **Merging (Sequential Step in Merge Sort)**  
   - Combines two sorted subarrays into one.  
   - Typically **not parallelized** due to dependencies.  

9. **Early Termination in Bubble Sort**  
   - If no swaps occur in a pass, the array is sorted (`swapped` flag optimization).  

10. **Shared vs. Private Variables**  
    - **Shared**: `arr`, `swapped` (accessed by all threads).  
    - **Private**: Loop indices (`i`, `j`, `k`).  

11. **Thread Safety**  
    - **`atomic write`** ensures only one thread updates `swapped` at a time.  

12. **Performance Trade-offs**  
    - **Bubble Sort (Parallel)**: Works best for small datasets due to high overhead.  
    - **Merge Sort (Parallel)**: Better for large datasets (scales well with threads).  

*/




