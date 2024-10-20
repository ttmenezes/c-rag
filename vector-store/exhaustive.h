#ifndef EXHAUSTIVE_H
#define EXHAUSTIVE_H

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

void init_exhaustive_store(ExhaustiveStore* store, int dimensions);
void insert_exhaustive(ExhaustiveStore* store, float* vector);
int search_exhaustive(ExhaustiveStore* store, float* query, int k, int* result, float* distances);
void print_exhaustive_stats(ExhaustiveStore* store);

#endif // EXHAUSTIVE_H
