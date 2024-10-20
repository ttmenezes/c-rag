#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

#include <stdlib.h>

typedef struct {
    int index;
    float distance;
} PQElement;

typedef struct {
    PQElement* elements;
    int capacity;
    int size;
} PriorityQueue;

void init_priority_queue(PriorityQueue* pq, int capacity);
void push_priority_queue(PriorityQueue* pq, int index, float distance);
PQElement pop_priority_queue(PriorityQueue* pq);
int is_priority_queue_empty(PriorityQueue* pq);
void clear_priority_queue(PriorityQueue* pq);

#endif // PRIORITY_QUEUE_H
