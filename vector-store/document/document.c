#include "document.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Function implementations...

void init_document_store(DocumentStore* store, int initial_capacity, int vector_dimensions) {
    store->documents = malloc(initial_capacity * sizeof(Document));
    store->count = 0;
    store->capacity = initial_capacity;
    store->vector_dim = vector_dimensions;
}

int add_document(DocumentStore* store, float* vector, const char* text) {
    if (store->count >= store->capacity) {
        // Resize logic here if needed
        return -1;
    }

    Document* doc = &store->documents[store->count];

    // Allocate memory for the vector and copy it
    doc->vector = (float*)malloc(store->vector_dim * sizeof(float));
    if (doc->vector == NULL) {
        return -1;
    }
    memcpy(doc->vector, vector, store->vector_dim * sizeof(float));

    // Allocate memory for the text and copy it
    doc->text = strdup(text);
    if (doc->text == NULL) {
        free(doc->vector);
        return -1;
    }

    doc->vector_dim = store->vector_dim;

    store->count++;
    return store->count - 1;
}

Document* get_document(DocumentStore* store, int id) {
    if (id < 0 || id >= store->count) {
        return NULL;
    }
    return &store->documents[id];
}

bool update_document(DocumentStore* store, int id, float* vector, const char* text) {
    if (id < 0 || id >= store->count) {
        return false;
    }
    
    Document* doc = &store->documents[id];
    memcpy(doc->vector, vector, store->vector_dim * sizeof(float));
    strncpy(doc->text, text, MAX_TEXT_LENGTH - 1);
    doc->text[MAX_TEXT_LENGTH - 1] = '\0';
    
    return true;
}

bool delete_document(DocumentStore* store, int id) {
    if (id < 0 || id >= store->count) {
        return false;
    }
    
    free(store->documents[id].vector);
    free(store->documents[id].text);
    
    if (id < store->count - 1) {
        store->documents[id] = store->documents[store->count - 1];
        // If you need to update the id, ensure the Document struct has an id field
    }
    
    store->count--;
    return true;
}

void free_document_store(DocumentStore* store) {
    for (int i = 0; i < store->count; i++) {
        free(store->documents[i].vector);
        free(store->documents[i].text);
    }
    free(store->documents);
    store->documents = NULL;
    store->count = 0;
    store->capacity = 0;
}
