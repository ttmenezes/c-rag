#ifndef DOCUMENT_H
#define DOCUMENT_H

#include <stdbool.h>

#define MAX_TEXT_LENGTH 1000

typedef struct {
    float* vector;
    char* text;  // Use a pointer for dynamic allocation
    int vector_dim;
    int id;  // Add this field if you need to track document IDs
} Document;

typedef struct {
    Document* documents;
    int count;          // Updated field name to match implementation
    int capacity;
    int vector_dim;     // Updated field name to match implementation
} DocumentStore;

void init_document_store(DocumentStore* store, int initial_capacity, int vector_dimensions);
int add_document(DocumentStore* store, float* vector, const char* text);
Document* get_document(DocumentStore* store, int id);
bool update_document(DocumentStore* store, int id, float* vector, const char* text);
bool delete_document(DocumentStore* store, int id);
void free_document_store(DocumentStore* store);

#endif // DOCUMENT_H
