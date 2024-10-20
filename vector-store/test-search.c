#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hnsw.h"
#include "exhaustive.h"

#define NUM_VECTORS 300
#define DIMENSIONS 30

// Helper function to generate a random float between 0 and 1
float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Helper function to generate a random vector
void generate_random_vector(float* vector, int dimensions) {
    for (int i = 0; i < dimensions; i++) {
        vector[i] = random_float() * 10.0f;  // Scale to 0-10 range
    }
}

// Update the print_vector function to handle NULL pointers
void print_vector(float* vector, int dimensions) {
    if (vector == NULL) {
        printf("[NULL]");
        return;
    }
    printf("[");
    for (int i = 0; i < dimensions; i++) {
        printf("%.2f", vector[i]);
        if (i < dimensions - 1) printf(", ");
    }
    printf("]");
}

int main() {
    srand(time(NULL));  // Initialize random seed

    HNSW* hnsw = (HNSW*)malloc(sizeof(HNSW));
    ExhaustiveStore exhaustive;

    init_hnsw(hnsw, DIMENSIONS);
    init_exhaustive_store(&exhaustive, DIMENSIONS);

    // Generate random vectors
    float vectors[NUM_VECTORS][DIMENSIONS];
    for (int i = 0; i < NUM_VECTORS; i++) {
        generate_random_vector(vectors[i], DIMENSIONS);
    }

    // Benchmark HNSW insertion
    clock_t start = clock();
    for (int i = 0; i < NUM_VECTORS; i++) {
        insert(hnsw, vectors[i]);
    }
    clock_t end = clock();
    double hnsw_insert_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark Exhaustive insertion
    start = clock();
    for (int i = 0; i < NUM_VECTORS; i++) {
        insert_exhaustive(&exhaustive, vectors[i]);
    }
    end = clock();
    double exhaustive_insert_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // print times for insertion
    // printf("HNSW insertion time: %.6f seconds\n", hnsw_insert_time);
    // printf("Exhaustive insertion time: %.6f seconds\n", exhaustive_insert_time);

    // print all nodes in hnsw
    print_all_nodes(hnsw);
    // Generate a random query vector
    float query[DIMENSIONS];
    generate_random_vector(query, DIMENSIONS);

    printf("Query vector: ");
    print_vector(query, DIMENSIONS);
    printf("\n\n");

    // Benchmark HNSW search
    int hnsw_result[10];
    float hnsw_distances[10];
    start = clock();
    int hnsw_num_results = search(hnsw, query, 10, hnsw_result, hnsw_distances);
    end = clock();
    double hnsw_search_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark Exhaustive search
    int exhaustive_result[10];
    float exhaustive_distances[10];
    start = clock();
    int exhaustive_num_results = search_exhaustive(&exhaustive, query, 10, exhaustive_result, exhaustive_distances);
    end = clock();
    double exhaustive_search_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print benchmark results
    printf("Benchmark Results:\n");
    printf("%-20s %-20s %-20s\n", "Operation", "HNSW Time (s)", "Exhaustive Time (s)");
    printf("%-20s %-20.6f %-20.6f\n", "Insertion", hnsw_insert_time, exhaustive_insert_time);
    printf("%-20s %-20.6f %-20.6f\n", "Search", hnsw_search_time, exhaustive_search_time);
    printf("\n");

    // Print search results
    printf("HNSW Search Results:\n");
    printf("%-10s %-10s %-s\n", "Index", "Distance", "Vector");
    for (int i = 0; i < hnsw_num_results; i++) {
        int index = hnsw_result[i];
        printf("%-10d %-10.4f ", index, hnsw_distances[i]);
        // take vector list from exhaustive store
        print_vector(exhaustive.elements[index].vector, DIMENSIONS);
        printf("\n");
    }

    printf("\nExhaustive Search Results:\n");
    printf("%-10s %-10s %-s\n", "Index", "Distance", "Vector");
    for (int i = 0; i < exhaustive_num_results; i++) {
        int index = exhaustive_result[i];
        printf("%-10d %-10.4f ", index, exhaustive_distances[i]);
        print_vector(exhaustive.elements[index].vector, DIMENSIONS);
        printf("\n");
    }

    // Free allocated memory
    free_hnsw(hnsw);

    return 0;
}
