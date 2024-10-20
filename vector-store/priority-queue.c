#include "priority-queue.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void swap(PQElement* a, PQElement* b) {
    PQElement temp = *a;
    *a = *b;
    *b = temp;
}

static void heapify_up(PriorityQueue* pq, int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (pq->elements[index].distance >= pq->elements[parent].distance) break;
        swap(&pq->elements[index], &pq->elements[parent]);
        index = parent;
    }
}

static void heapify_down(PriorityQueue* pq, int index) {
    int min_index = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < pq->size && pq->elements[left].distance < pq->elements[min_index].distance)
        min_index = left;
    if (right < pq->size && pq->elements[right].distance < pq->elements[min_index].distance)
        min_index = right;

    if (index != min_index) {
        swap(&pq->elements[index], &pq->elements[min_index]);
        heapify_down(pq, min_index);
    }
}

void init_priority_queue(PriorityQueue* pq, int capacity) {
    pq->capacity = capacity;
    pq->size = 0;
    pq->elements = (PQElement*)malloc(capacity * sizeof(PQElement));
    if (pq->elements == NULL) {
        fprintf(stderr, "Failed to allocate memory for priority queue elements\n");
        exit(1);
    }
}

void push_priority_queue(PriorityQueue* pq, int index, float distance) {
    if (pq->size == pq->capacity) {
        // Double the capacity
        pq->capacity *= 2;
        pq->elements = realloc(pq->elements, pq->capacity * sizeof(PQElement));
        if (pq->elements == NULL) {
            fprintf(stderr, "Failed to reallocate memory for priority queue\n");
            exit(1);
        }
    }
    pq->elements[pq->size] = (PQElement){index, distance};
    heapify_up(pq, pq->size);
    pq->size++;
}

PQElement pop_priority_queue(PriorityQueue* pq) {
    if (pq->size == 0) return (PQElement){-1, 0.0f};
    PQElement top = pq->elements[0];
    pq->size--;
    pq->elements[0] = pq->elements[pq->size];
    heapify_down(pq, 0);
    return top;
}

int is_priority_queue_empty(PriorityQueue* pq) {
    return pq->size == 0;
}

void clear_priority_queue(PriorityQueue* pq) {
    // Reset the size to 0
    pq->size = 0;
    // Clear the memory of the elements array
    memset(pq->elements, 0, pq->capacity * sizeof(PQElement));
}
