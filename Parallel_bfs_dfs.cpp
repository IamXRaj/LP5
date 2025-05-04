#include <iostream>       // For input/output operations
#include <vector>         // For using vector container
#include <queue>          // For BFS queue implementation
#include <omp.h>          // OpenMP library for parallel programming
#include <stack>          // For DFS stack implementation

using namespace std; // Use the entire std namespace

// Graph class representing an undirected graph using adjacency lists
class Graph {
    int V;  // Number of vertices in the graph
    vector<vector<int>> adj;  // Adjacency list representation

public:
    // Constructor to initialize graph with V vertices
    Graph(int v) : V(v), adj(v) {}  // Initialize V and resize adjacency list to v

    // Method to add an edge between vertices v and w
    void addEdge(int v, int w) { adj[v].push_back(w); }  // Add w to v's adjacency list

    // Parallel Breadth-First Search implementation
    void parallelBFS(int start) {
        vector<bool> visited(V, false);  // Track visited vertices
        queue<int> q;                   // Queue for BFS traversal
        
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
                    cout << v << " ";  // Process current vertex
                    
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
        cout << "\n";  // Newline after traversal
    }

    // Parallel Depth-First Search implementation
    void parallelDFS(int start) {
        vector<bool> visited(V, false);  // Track visited vertices
        
        #pragma omp parallel  // Start parallel region
        {
            stack<int> s;  // Each thread gets its own stack
            
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
                        cout << v << " "; // Process current vertex
                        
                        // Push adjacent vertices in reverse order (for DFS)
                        for (auto it = adj[v].rbegin(); it != adj[v].rend(); ++it)
                            s.push(*it);
                    }
                }
            }
        }
        cout << "\n";  // Newline after traversal
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
    cout << "BFS: ";
    g.parallelBFS(0);
    
    // Perform and print DFS traversal starting from vertex 0
    cout << "DFS: ";
    g.parallelDFS(0);
    
    return 0;  // End of program
}
