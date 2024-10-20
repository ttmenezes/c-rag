#ifndef HNSW_H
#define HNSW_H

#include <stdbool.h>
#include "priority-queue.h"

#define MAX_ELEMENTS 10000
#define MAX_DIMENSIONS 128
#define MAX_LEVELS 16
#define M 16
#define ef_construction 200
#define PQ_SIZE 500

typedef struct Node {
    float vector[MAX_DIMENSIONS];
    int connections[MAX_LEVELS][M];
    int num_connections[MAX_LEVELS];
    int level;
} Node;

typedef struct HNSW {
    Node nodes[MAX_ELEMENTS];
    int num_elements;
    int max_level;
    int dimensions;
    PriorityQueue* level_pqs;
} HNSW;

void init_hnsw(HNSW* hnsw, int dimensions);
void insert(HNSW* hnsw, float* vector);
int search(HNSW* hnsw, float* query, int k, int* result, float* distances);
void print_hnsw_stats(HNSW* hnsw);
void free_hnsw(HNSW* hnsw);
void print_all_nodes(HNSW* hnsw);
void select_neighbors(HNSW* hnsw, int current, PriorityQueue* candidates, int level, int max_connections);

#endif // HNSW_H
