#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <string.h>
#include "priority-queue.h"  // Include the priority queue header
#include "util.h"

#define MAX_ELEMENTS 10000
#define MAX_DIMENSIONS 128
#define MAX_LEVELS 16
#define M 16 // Maximum number of connections per node at each level
#define ef_construction 200  // Increase from 100 to 200 or higher
#define PQ_SIZE 500  // Or any other suitable value
#define ef_search 150  // Reduce from 50 to 30

// Move contains_priority_queue declaration and implementation here
bool contains_priority_queue(PriorityQueue* pq, int index) {
    for (int i = 0; i < pq->size; i++) {
        if (pq->elements[i].index == index) {
            return true;
        }
    }
    return false;
}

// Add this helper function
bool contains_connection(int* connections, int num_connections, int index) {
    for (int i = 0; i < num_connections; i++) {
        if (connections[i] == index) {
            return true;
        }
    }
    return false;
}

typedef struct Node {
    float vector[MAX_DIMENSIONS];
    int connections[MAX_LEVELS][M];
    int num_connections[MAX_LEVELS];  // Track the actual number of connections at each level
    int level;
} Node;

typedef struct HNSW {
    Node nodes[MAX_ELEMENTS];
    int num_elements;
    int max_level;
    int dimensions;
    PriorityQueue* level_pqs;  // Add this line
} HNSW;

void init_hnsw(HNSW* hnsw, int dimensions) {
    hnsw->num_elements = 0;
    hnsw->max_level = 0;
    hnsw->dimensions = dimensions;
    
    hnsw->level_pqs = malloc(MAX_LEVELS * sizeof(PriorityQueue));
    if (hnsw->level_pqs == NULL) {
        fprintf(stderr, "Failed to allocate memory for level priority queues\n");
        exit(1);
    }
    for (int i = 0; i < MAX_LEVELS; i++) {
        init_priority_queue(&hnsw->level_pqs[i], PQ_SIZE);  // Use PQ_SIZE instead of ef_construction
    }
}

int get_random_level() {
    float r = ((float)rand() / (float)RAND_MAX);
    return (int)(-log(r) * (1.0 / log(4)));  // Change base from 2 to 4 to reduce max level
}

// New helper function to select neighbors
void select_neighbors(HNSW* hnsw, int current, PriorityQueue* candidates, int level, int max_connections) {
    PriorityQueue temp_queue;
    init_priority_queue(&temp_queue, candidates->size);

    while (!is_priority_queue_empty(candidates)) {
        PQElement element = pop_priority_queue(candidates);
        push_priority_queue(&temp_queue, element.index, -element.distance);  // Note the negation here
    }

    int added = 0;
    while (!is_priority_queue_empty(&temp_queue) && added < max_connections) {
        PQElement best = pop_priority_queue(&temp_queue);
        
        // Check if connection already exists
        bool exists = false;
        for (int i = 0; i < hnsw->nodes[current].num_connections[level]; i++) {
            if (hnsw->nodes[current].connections[level][i] == best.index) {
                exists = true;
                break;
            }
        }
        
        if (!exists) {
            hnsw->nodes[current].connections[level][hnsw->nodes[current].num_connections[level]++] = best.index;
            added++;
        }
    }

    clear_priority_queue(&temp_queue);
}

// Updated insert function
void insert(HNSW* hnsw, float* vector) {
    if (hnsw->num_elements >= MAX_ELEMENTS) {
        printf("HNSW is full\n");
        return;
    }

    int new_element_index = hnsw->num_elements;
    Node* new_element = &hnsw->nodes[new_element_index];

    memcpy(new_element->vector, vector, hnsw->dimensions * sizeof(float));
    memset(new_element->num_connections, 0, sizeof(new_element->num_connections));

    new_element->level = get_random_level();
    if (new_element->level > hnsw->max_level) {
        hnsw->max_level = new_element->level;
    }

    if (hnsw->num_elements == 0) {
        hnsw->num_elements++;
        return;
    }

    int entry_point = 0;  // Start with the first element as entry point

    for (int current_level = hnsw->max_level; current_level >= 0; current_level--) {
        PriorityQueue candidates;
        init_priority_queue(&candidates, ef_construction);

        float entry_dist = euclidean_distance(vector, hnsw->nodes[entry_point].vector, hnsw->dimensions);
        push_priority_queue(&candidates, entry_point, entry_dist);

        // Search for ef_construction nearest neighbors
        PriorityQueue visited;
        init_priority_queue(&visited, ef_construction);

        while (!is_priority_queue_empty(&candidates)) {
            PQElement current = pop_priority_queue(&candidates);
            push_priority_queue(&visited, current.index, current.distance);

            if (current.distance > visited.elements[visited.size - 1].distance && visited.size >= ef_construction) break;

            for (int i = 0; i < hnsw->nodes[current.index].num_connections[current_level]; i++) {
                int neighbor = hnsw->nodes[current.index].connections[current_level][i];
                if (!contains_priority_queue(&visited, neighbor)) {
                    float dist = euclidean_distance(vector, hnsw->nodes[neighbor].vector, hnsw->dimensions);
                    if (visited.size < ef_construction || dist < visited.elements[visited.size - 1].distance) {
                        push_priority_queue(&candidates, neighbor, dist);
                    }
                }
            }
        }

        // Connect the new element to its nearest neighbors at this level
        if (current_level <= new_element->level) {
            while (!is_priority_queue_empty(&visited) && new_element->num_connections[current_level] < M) {
                PQElement neighbor = pop_priority_queue(&visited);
                
                if (!contains_connection(new_element->connections[current_level], new_element->num_connections[current_level], neighbor.index)) {
                    new_element->connections[current_level][new_element->num_connections[current_level]++] = neighbor.index;
                
                    // Add bidirectional connection
                    if (hnsw->nodes[neighbor.index].num_connections[current_level] < M) {
                        if (!contains_connection(hnsw->nodes[neighbor.index].connections[current_level], 
                                                 hnsw->nodes[neighbor.index].num_connections[current_level], 
                                                 new_element_index)) {
                            hnsw->nodes[neighbor.index].connections[current_level][hnsw->nodes[neighbor.index].num_connections[current_level]++] = new_element_index;
                        }
                    } else {
                        // Replace the farthest connection if the new element is closer
                        int farthest_index = -1;
                        float max_dist = -1;
                        for (int j = 0; j < M; j++) {
                            int existing = hnsw->nodes[neighbor.index].connections[current_level][j];
                            float existing_dist = euclidean_distance(hnsw->nodes[neighbor.index].vector, hnsw->nodes[existing].vector, hnsw->dimensions);
                            if (existing_dist > max_dist) {
                                max_dist = existing_dist;
                                farthest_index = j;
                            }
                        }
                        if (neighbor.distance < max_dist && !contains_connection(hnsw->nodes[neighbor.index].connections[current_level], M, new_element_index)) {
                            hnsw->nodes[neighbor.index].connections[current_level][farthest_index] = new_element_index;
                        }
                    }
                }
            }
        }

        // Update entry point for the next level
        if (!is_priority_queue_empty(&visited)) {
            entry_point = visited.elements[0].index;  // The closest element
        }

        clear_priority_queue(&visited);
        clear_priority_queue(&candidates);
    }

    hnsw->num_elements++;
}

void print_all_nodes(HNSW* hnsw) {
    printf("HNSW Nodes:\n");
    for (int i = 0; i < hnsw->num_elements; i++) {
        Node* node = &hnsw->nodes[i];
        printf("Node %d (Level %d):\n", i, node->level);
        
        printf("  Vector: [");
        for (int j = 0; j < hnsw->dimensions; j++) {
            printf("%.2f", node->vector[j]);
            if (j < hnsw->dimensions - 1) printf(", ");
        }
        printf("]\n");
        
        printf("  Connections:\n");
        for (int level = 0; level <= node->level; level++) {
            printf("    Level %d: [", level);
            for (int j = 0; j < node->num_connections[level]; j++) {
                printf("%d", node->connections[level][j]);
                if (j < node->num_connections[level] - 1) printf(", ");
            }
            printf("]\n");
        }
        printf("\n");
    }
}

// Declare search_layer right before search function
void search_layer(HNSW* hnsw, float* query, int* ep, int level, PriorityQueue* candidates);

int search(HNSW* hnsw, float* query, int k, int* result, float* distances) {
    PriorityQueue candidates;
    init_priority_queue(&candidates, ef_search);

    // printf("Starting search for query: [");
    // for (int i = 0; i < hnsw->dimensions; i++) {
    //     printf("%.2f", query[i]);
    //     if (i < hnsw->dimensions - 1) printf(", ");
    // }
    // printf("]\n");
    // printf("Number of elements in HNSW: %d\n", hnsw->num_elements);

    int ep = 0;  // entry point
    for (int level = hnsw->max_level; level >= 0; level--) {
        // printf("Searching level %d\n", level);
        search_layer(hnsw, query, &ep, level, &candidates);
    }

    // printf("Candidates after search: %d\n", candidates.size);

    // Get top k results
    PriorityQueue results;
    init_priority_queue(&results, k);

    while (!is_priority_queue_empty(&candidates)) {
        PQElement element = pop_priority_queue(&candidates);
        if (results.size < k) {
            push_priority_queue(&results, element.index, element.distance);
        } else if (element.distance < results.elements[0].distance) {
            pop_priority_queue(&results);
            push_priority_queue(&results, element.index, element.distance);
        }
    }

    int num_results = results.size;
    for (int i = 0; i < num_results; i++) {
        PQElement element = pop_priority_queue(&results);
        result[i] = element.index;
        distances[i] = element.distance;
        // printf("Result %d: index %d, distance %.4f\n", i, result[i], distances[i]);
    }

    clear_priority_queue(&results);
    clear_priority_queue(&candidates);
    return num_results;
}

void search_layer(HNSW* hnsw, float* query, int* ep, int level, PriorityQueue* candidates) {
    PriorityQueue visited;
    init_priority_queue(&visited, ef_search);

    // Add a boolean array to keep track of visited nodes
    bool* node_visited = calloc(hnsw->num_elements, sizeof(bool));
    if (node_visited == NULL) {
        fprintf(stderr, "Failed to allocate memory for node_visited array\n");
        exit(1);
    }

    float dist = euclidean_distance(query, hnsw->nodes[*ep].vector, hnsw->dimensions);
    push_priority_queue(candidates, *ep, dist);
    push_priority_queue(&visited, *ep, dist);
    node_visited[*ep] = true;

    int iterations = 0;
    int max_iterations = hnsw->num_elements * 2;  // Set a reasonable upper limit

    while (!is_priority_queue_empty(candidates) && iterations < max_iterations) {
        iterations++;
        PQElement current = pop_priority_queue(candidates);
        
        if (current.distance > visited.elements[visited.size - 1].distance && visited.size >= ef_search) {
            break;
        }

        // printf("Iteration %d: Exploring node %d at level %d (distance: %.4f)\n", iterations, current.index, level, current.distance);

        int neighbors_explored = 0;
        for (int i = 0; i < hnsw->nodes[current.index].num_connections[level]; i++) {
            int neighbor = hnsw->nodes[current.index].connections[level][i];
            if (!node_visited[neighbor]) {
                node_visited[neighbor] = true;
                dist = euclidean_distance(query, hnsw->nodes[neighbor].vector, hnsw->dimensions);
                if (visited.size < ef_search || dist < visited.elements[visited.size - 1].distance) {
                    push_priority_queue(candidates, neighbor, dist);
                    push_priority_queue(&visited, neighbor, dist);
                    if (visited.size > ef_search) {
                        pop_priority_queue(&visited);
                    }
                    neighbors_explored++;
                }
            }
        }
        // printf("  Explored %d neighbors\n", neighbors_explored);

        // Early exit condition
        if (iterations > 100 && neighbors_explored == 0) {
            // printf("Early exit: No new neighbors found after %d iterations\n", iterations);
            break;
        }
    }

    if (iterations >= max_iterations) {
        printf("Warning: Search layer reached maximum iterations (%d) at level %d\n", max_iterations, level);
    }

    clear_priority_queue(candidates);
    while (!is_priority_queue_empty(&visited)) {
        push_priority_queue(candidates, visited.elements[0].index, visited.elements[0].distance);
        pop_priority_queue(&visited);
    }
    clear_priority_queue(&visited);

    *ep = candidates->elements[0].index;  // Update to the closest point
    // printf("Finished search_layer at level %d after %d iterations. New entry point: %d\n", level, iterations, *ep);

    free(node_visited);
}

void print_hnsw_stats(HNSW* hnsw) {
    printf("HNSW Stats:\n");
    printf("Number of elements: %d\n", hnsw->num_elements);
    printf("Maximum level: %d\n", hnsw->max_level);
    printf("Dimensions: %d\n", hnsw->dimensions);
}

// Add a function to free the HNSW structure
void free_hnsw(HNSW* hnsw) {
    for (int i = 0; i < MAX_LEVELS; i++) {
        free(hnsw->level_pqs[i].elements);
    }
    free(hnsw->level_pqs);
    free(hnsw);
}

void verify_graph_structure(HNSW* hnsw) {
    for (int i = 0; i < hnsw->num_elements; i++) {
        for (int level = 0; level <= hnsw->nodes[i].level; level++) {
            for (int j = 0; j < hnsw->nodes[i].num_connections[level]; j++) {
                int neighbor = hnsw->nodes[i].connections[level][j];
                if (neighbor >= hnsw->num_elements) {
                    printf("Error: Node %d at level %d has invalid connection to %d\n", i, level, neighbor);
                }
                if (!contains_connection(hnsw->nodes[neighbor].connections[level], hnsw->nodes[neighbor].num_connections[level], i)) {
                    printf("Warning: Bidirectional connection missing between %d and %d at level %d\n", i, neighbor, level);
                }
            }
        }
    }
    printf("Graph structure verification complete.\n");
}

// int main() {
//     HNSW* hnsw = (HNSW*)malloc(sizeof(HNSW));
//     if (hnsw == NULL) {
//         fprintf(stderr, "Failed to allocate memory for HNSW\n");
//         return 1;
//     }

//     init_hnsw(hnsw, 3);

//     // Insert sample vectors
//     float vectors[][3] = {
//         {1.0, 2.0, 3.0}, {1.0, 2.0, 3.5}, {1.0, 2.1, 3.0},
//         {2.5, 3.5, 4.5}, {5.5, 6.5, 7.5}, {1.0, 2.0, 3.1},
//         {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}, {1.5, 2.5, 3.5},
//         {4.5, 5.5, 6.5}, {7.5, 8.5, 9.5}, {0.1, 0.2, 0.3}
//     };

//     for (int i = 0; i < 12; i++) {
//         insert(hnsw, vectors[i]);
//         printf("Inserted vector %d: [%.2f, %.2f, %.2f]\n", i, vectors[i][0], vectors[i][1], vectors[i][2]);
//         printf("Current number of elements: %d\n", hnsw->num_elements);
//     }

//     print_hnsw_stats(hnsw);
//     print_all_nodes(hnsw);

//     // search for a vector
//     float query[] = {0.2, 0.2, 0.3};
//     int result[10];
//     float distances[10];
//     int num_results = search(hnsw, query, 10, result, distances);
    
//     printf("Search results:\n");
//     printf("Query vector: [%.2f, %.2f, %.2f]\n", query[0], query[1], query[2]);
//     printf("%-10s %-10s %-s\n", "Index", "Distance", "Vector");
//     for (int i = 0; i < num_results; i++) {
//         int index = result[i];
//         float* vector = hnsw->nodes[index].vector;
//         printf("%-10d %-10.4f [%.2f, %.2f, %.2f]\n", 
//                index, distances[i], vector[0], vector[1], vector[2]);
//     }

//     free_hnsw(hnsw);
//     return 0;
// }

