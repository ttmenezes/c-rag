#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "./vector-store/hnsw.h"
#include "./vector-store/exhaustive.h"
#include "./vector-store/document/document.h"
#include "embedding-model/embedding_model.h"  // Include the embedding model header

#define MAX_SENTENCES 30
#define MAX_TEXT_LENGTH 1000

// New function to read sentences from file
char** read_sentences(const char* filename, int* num_sentences) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    char** sentences = malloc(MAX_SENTENCES * sizeof(char*));
    char buffer[MAX_TEXT_LENGTH];
    int count = 0;

    while (fgets(buffer, MAX_TEXT_LENGTH, file) && count < MAX_SENTENCES) {
        // Remove newline character
        buffer[strcspn(buffer, "\n")] = 0;
        sentences[count] = strdup(buffer);
        count++;
    }

    fclose(file);
    *num_sentences = count;
    return sentences;
}

void print_vector(float* vector, int dimensions) {
    printf("[");
    for (int i = 0; i < dimensions; i++) {
        printf("%.4f", vector[i]);
        if (i < dimensions - 1) printf(", ");
    }
    printf("]");
}

float calculate_checksum(float* vector, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    const char* sentences_file = "sentences.txt";
    int num_sentences;
    char** sentences = read_sentences(sentences_file, &num_sentences);
    if (!sentences) {
        fprintf(stderr, "Failed to read sentences\n");
        return 1;
    }

    DocumentStore doc_store;
    init_document_store(&doc_store, num_sentences, EMBEDDING_DIM);

    HNSW* hnsw = (HNSW*)malloc(sizeof(HNSW));
    if (!hnsw) {
        fprintf(stderr, "Failed to allocate memory for HNSW\n");
        // Clean up and exit
        return 1;
    }
    ExhaustiveStore exhaustive;

    init_hnsw(hnsw, EMBEDDING_DIM);
    init_exhaustive_store(&exhaustive, EMBEDDING_DIM);
    
    const char* vocab_file = "./embedding-model/vocab.txt";
    const char* model_file = "./embedding-model/model.bin";

    // Embed sentences and insert into indexes
    for (int i = 0; i < num_sentences; i++) {
        printf("Embedding document %d: %s\n", i, sentences[i]);
        float* vector = embed_text(sentences[i], vocab_file, model_file);
        if (!vector) {
            fprintf(stderr, "Failed to embed text for document %d\n", i);
            continue;
        }

        printf("Successfully embedded document %d\n", i);

        float checksum_before = calculate_checksum(vector, EMBEDDING_DIM);
        printf("Checksum before operations: %.4f\n", checksum_before);

        int doc_id = add_document(&doc_store, vector, sentences[i]);
        printf("Added document to store with ID %d\n", doc_id);

        insert(hnsw, vector);
        printf("Inserted into HNSW index\n");

        insert_exhaustive(&exhaustive, vector);
        printf("Inserted into exhaustive index\n");

        float checksum_after = calculate_checksum(vector, EMBEDDING_DIM);
        printf("Checksum after operations: %.4f\n", checksum_after);

        if (checksum_before != checksum_after) {
            fprintf(stderr, "Warning: Vector checksum changed for document %d\n", i);
        }

        printf("Added document %d: %s\n", doc_id, sentences[i]);
        // Remove the free(vector) call
    }

    printf("\nDocument store and indexes populated.\n\n");

    // Use command-line argument as query if provided, otherwise use default
    char* query_text;
    if (argc > 1) {
        query_text = argv[1];
    } else {
        query_text = "brazil world cup";
    }

    float* query_vector = embed_text(query_text, vocab_file, model_file);
    if (!query_vector) {
        fprintf(stderr, "Failed to embed query text\n");
        // Clean up and exit
        // ... (free other resources)
        return 1;
    }

    printf("Query text: %s\n", query_text);
    printf("Query vector: ");
    print_vector(query_vector, EMBEDDING_DIM);
    printf("\n\n");

    // Search using HNSW
    int hnsw_result[10];
    float hnsw_distances[10];
    int hnsw_num_results = search(hnsw, query_vector, 10, hnsw_result, hnsw_distances);

    // Search using Exhaustive
    int exhaustive_result[10];
    float exhaustive_distances[10];
    int exhaustive_num_results = search_exhaustive(&exhaustive, query_vector, 10, exhaustive_result, exhaustive_distances);

    // Print HNSW results
    printf("HNSW Search Results:\n");
    printf("%-10s %-10s %-30s %-s\n", "Doc ID", "Distance", "Text (truncated)", "Vector (first 5 values)");
    for (int i = 0; i < hnsw_num_results; i++) {
        int doc_id = hnsw_result[i];
        Document* doc = get_document(&doc_store, doc_id);
        printf("%-10d %-10.4f %-30.30s [", doc_id, hnsw_distances[i], doc->text);
        for (int j = 0; j < 5; j++) {
            printf("%.4f", doc->vector[j]);
            if (j < 4) printf(", ");
        }
        printf("...]\n");
    }

    // Print Exhaustive search results
    printf("\nExhaustive Search Results:\n");
    printf("%-10s %-10s %-30s %-s\n", "Doc ID", "Distance", "Text (truncated)", "Vector (first 5 values)");
    for (int i = 0; i < exhaustive_num_results; i++) {
        int doc_id = exhaustive_result[i];
        Document* doc = get_document(&doc_store, doc_id);
        printf("%-10d %-10.4f %-30.30s [", doc_id, exhaustive_distances[i], doc->text);
        for (int j = 0; j < 5; j++) {
            printf("%.4f", doc->vector[j]);
            if (j < 4) printf(", ");
        }
        printf("...]\n");
    }

    // Free allocated memory
    free_document_store(&doc_store);
    free_hnsw(hnsw);
    // free(hnsw);
    // Ensure exhaustive is freed correctly
    // free_exhaustive_store(&exhaustive); // Uncomment if you have a function to free exhaustive

    // for (int i = 0; i < num_sentences; i++) {
    //     free(sentences[i]);  // Ensure each strdup'd string is freed
    // }
    // free(sentences);  // Free the array of pointers

    // free(query_vector);  // Ensure the query vector is freed

    return 0;
}
