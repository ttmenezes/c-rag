#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "util.h"

#define MAX_ELEMENTS 10000
#define MAX_DIMENSIONS 128

typedef struct {
    float vector[MAX_DIMENSIONS];
} Element;

typedef struct {
    Element elements[MAX_ELEMENTS];
    int num_elements;
    int dimensions;
} ExhaustiveStore;

void init_exhaustive_store(ExhaustiveStore* store, int dimensions) {
    store->num_elements = 0;
    store->dimensions = dimensions;
}

void insert_exhaustive(ExhaustiveStore* store, float* vector) {
    if (store->num_elements >= MAX_ELEMENTS) {
        printf("ExhaustiveStore is full\n");
        return;
    }

    Element* new_element = &store->elements[store->num_elements];
    memcpy(new_element->vector, vector, store->dimensions * sizeof(float));
    store->num_elements++;
}

int search_exhaustive(ExhaustiveStore* store, float* query, int k, int* result, float* distances) {
    // Allocate memory for temporary arrays
    float* temp_distances = malloc(store->num_elements * sizeof(float));
    int* temp_result = malloc(store->num_elements * sizeof(int));

    if (!temp_distances || !temp_result) {
        fprintf(stderr, "Memory allocation failed in search_exhaustive\n");
        exit(1);
    }

    for (int i = 0; i < store->num_elements; i++) {
        temp_distances[i] = euclidean_distance(query, store->elements[i].vector, store->dimensions);
        temp_result[i] = i;
    }

    // Simple bubble sort to get top k results
    for (int i = 0; i < k && i < store->num_elements; i++) {
        for (int j = i + 1; j < store->num_elements; j++) {
            if (temp_distances[j] < temp_distances[i]) {
                float temp_dist = temp_distances[i];
                temp_distances[i] = temp_distances[j];
                temp_distances[j] = temp_dist;

                int temp_index = temp_result[i];
                temp_result[i] = temp_result[j];
                temp_result[j] = temp_index;
            }
        }
    }

    // Copy results to output arrays
    int num_results = (store->num_elements < k) ? store->num_elements : k;
    memcpy(result, temp_result, num_results * sizeof(int));
    memcpy(distances, temp_distances, num_results * sizeof(float));

    // Free temporary arrays
    free(temp_distances);
    free(temp_result);

    return num_results;
}

void print_exhaustive_stats(ExhaustiveStore* store) {
    printf("ExhaustiveStore Stats:\n");
    printf("Number of elements: %d\n", store->num_elements);
    printf("Dimensions: %d\n", store->dimensions);
}

// int main() {
//     ExhaustiveStore store;
//     init_exhaustive_store(&store, 3);

//     // Insert sample vectors (same as in HNSW example)
//     float vectors[][3] = {
//         {1.0, 2.0, 3.0}, {1.0, 2.0, 3.5}, {1.0, 2.1, 3.0},
//         {2.5, 3.5, 4.5}, {5.5, 6.5, 7.5}, {1.0, 2.0, 3.1},
//         {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}, {1.5, 2.5, 3.5},
//         {4.5, 5.5, 6.5}, {7.5, 8.5, 9.5}, {0.1, 0.2, 0.3}
//     };

//     for (int i = 0; i < 12; i++) {
//         insert_exhaustive(&store, vectors[i]);
//     }

//     // print_exhaustive_stats(&store);

//     // Search for a vector
//     float query[] = {3.0, 4.0, 5.1};  // Updated to match your output
//     int result[10];
//     float distances[10];
//     int num_results = search_exhaustive(&store, query, 10, result, distances);

//     printf("Search results:\n");
//     printf("Query vector: [%.2f, %.2f, %.2f]\n", query[0], query[1], query[2]);
//     printf("%-10s %-10s %-s\n", "Index", "Distance", "Vector");
//     for (int i = 0; i < num_results; i++) {
//         int index = result[i];
//         float* vector = store.elements[index].vector;
//         printf("%-10d %-10.4f [%.2f, %.2f, %.2f]\n", 
//                index, distances[i], vector[0], vector[1], vector[2]);
//     }

//     return 0;
// }
