#include <iostream>       // For input/output operations
#include <vector>         // For using vector container
#include <queue>          // For BFS queue implementation
#include <omp.h>          // OpenMP library for parallel programming
#include <stack>          // For DFS stack implementation

// Graph class representing an undirected graph using adjacency lists
class Graph {
    int V;  // Number of vertices in the graph
    std::vector<std::vector<int>> adj;  // Adjacency list representation

public:
    // Constructor to initialize graph with V vertices
    Graph(int v) : V(v), adj(v) {}  // Initialize V and resize adjacency list to v

    // Method to add an edge between vertices v and w
    void addEdge(int v, int w) { adj[v].push_back(w); }  // Add w to v's adjacency list

    // Parallel Breadth-First Search implementation
    void parallelBFS(int start) {
        std::vector<bool> visited(V, false);  // Track visited vertices
        std::queue<int> q;                   // Queue for BFS traversal
        
        visited[start] = true;  // Mark start vertex as visited
        q.push(start);          // Enqueue start vertex

        while (!q.empty()) {
            #pragma omp parallel  // Start parallel region
            {
                #pragma omp for nowait  // Parallelize loop with no implicit barrier
                for (int i = 0; i < q.size(); i++) {
                    int v;
                    #pragma omp critical  // Critical section to prevent race condition
                    {
                        v = q.front();  // Get front element
                        q.pop();       // Remove front element
                    }
                    std::cout << v << " ";  // Process current vertex
                    
                    // Visit all adjacent vertices
                    for (int u : adj[v]) {
                        #pragma omp critical  // Critical section for shared variables
                        if (!visited[u]) {
                            visited[u] = true;  // Mark as visited
                            q.push(u);         // Enqueue adjacent vertex
                        }
                    }
                }
            }
        }
        std::cout << "\n";  // Newline after traversal
    }

    // Parallel Depth-First Search implementation
    void parallelDFS(int start) {
        std::vector<bool> visited(V, false);  // Track visited vertices
        
        #pragma omp parallel  // Start parallel region
        {
            std::stack<int> s;  // Each thread gets its own stack
            
            #pragma omp single  // Single thread executes this block
            s.push(start);      // Push start vertex to stack
            
            while (!s.empty()) {
                int v = -1;  // Initialize with invalid vertex
                
                #pragma omp critical  // Critical section for stack operations
                if (!s.empty()) {
                    v = s.top();  // Get top element
                    s.pop();     // Remove top element
                }
                
                if (v == -1) continue;  // Skip if no vertex was popped
                
                if (!visited[v]) {
                    #pragma omp critical  // Critical section for shared variables
                    if (!visited[v]) {    // Double-check pattern to prevent race
                        visited[v] = true;      // Mark as visited
                        std::cout << v << " "; // Process current vertex
                        
                        // Push adjacent vertices in reverse order (for DFS)
                        for (auto it = adj[v].rbegin(); it != adj[v].rend(); ++it)
                            s.push(*it);
                    }
                }
            }
        }
        std::cout << "\n";  // Newline after traversal
    }
};

int main() {
    // Create a graph with 4 vertices
    Graph g(4);
    
    // Add edges to the graph
    g.addEdge(0, 1);  // Edge 0->1
    g.addEdge(0, 2);  // Edge 0->2
    g.addEdge(1, 3);  // Edge 1->3
    g.addEdge(2, 3);  // Edge 2->3
    
    // Perform and print BFS traversal starting from vertex 0
    std::cout << "BFS: ";
    g.parallelBFS(0);
    
    // Perform and print DFS traversal starting from vertex 0
    std::cout << "DFS: ";
    g.parallelDFS(0);
    
    return 0;  // End of program
}


/*
### **Key Terms Related to the Code:**  

1. **Graph (Adjacency List)**  
   - Represents a graph using a list of connected vertices for each node.  
   - Efficient for sparse graphs (few edges).  

2. **BFS (Breadth-First Search)**  
   - Explores all neighbor nodes at the present level before moving deeper.  
   - Uses a **queue** for traversal.  

3. **DFS (Depth-First Search)**  
   - Explores as far as possible along each branch before backtracking.  
   - Uses a **stack** (explicit or implicit via recursion).  

4. **OpenMP**  
   - API for **parallel programming** in C/C++/Fortran.  
   - Uses compiler directives (`#pragma omp`) for multi-threading.  

5. **Parallel BFS/DFS**  
   - Uses multiple threads to speed up traversal.  
   - Requires **critical sections** (`#pragma omp critical`) to avoid race conditions.  

6. **Critical Section**  
   - A code block that only **one thread can execute at a time** to prevent data races.  

7. **Race Condition**  
   - When multiple threads access shared data simultaneously, leading to unpredictable results.  

8. **`#pragma omp parallel`**  
   - Creates a team of threads to execute code in parallel.  

9. **`#pragma omp for`**  
   - Splits loop iterations among threads for parallel execution.  

10. **`#pragma omp single`**  
    - Only **one thread** executes the block (used for initialization).  

11. **`nowait` Clause**  
    - Removes the implicit barrier after a parallel loop, allowing threads to proceed without waiting.  

12. **Double-Check Pattern**  
    - Used in parallel DFS to avoid race conditions when checking `visited[v]`.  

13. **Synchronization**  
    - Ensures threads coordinate properly (e.g., using `critical` sections).  

14. **Shared vs. Private Variables**  
    - **Shared**: Accessible by all threads (e.g., `visited`, `queue`, `stack`).  
    - **Private**: Each thread has its own copy (e.g., loop variables).  

15. **Traversal Order**  
    - **BFS**: Level-by-level (shortest path in unweighted graphs).  
    - **DFS**: Goes deep first (used in maze solving, topological sorting).  

This code demonstrates **parallel graph traversal** using **OpenMP** for better performance on multi-core systems.
*/