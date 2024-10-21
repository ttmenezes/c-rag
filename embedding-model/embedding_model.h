#ifndef EMBEDDING_MODEL_H
#define EMBEDDING_MODEL_H

#include <stdint.h>

#define EMBEDDING_DIM 384
#define VOCAB_SIZE 30522
#define MAX_SEQ_LENGTH 512
#define NUM_HIDDEN_LAYERS 6
#define NUM_ATTENTION_HEADS 12
#define INTERMEDIATE_SIZE 1536

// Tokenizer structure
typedef struct {
    char** vocab;
    int vocab_size;
} Tokenizer;

// Attention weights structure
typedef struct {
    float* query;
    float* key;
    float* value;
    float* output;
} AttentionWeights;

// FFN weights structure
typedef struct {
    float* intermediate;
    float* output;
} FFNWeights;

// Model structure
typedef struct {
    float* token_embeddings;
    float* position_embeddings;
    float* token_type_embeddings;
    float* embeddings_layer_norm_weight;
    float* embeddings_layer_norm_bias;
    
    struct {
        AttentionWeights attention;
        float* attention_layer_norm_weight;
        float* attention_layer_norm_bias;
        FFNWeights ffn;
        float* ffn_layer_norm_weight;
        float* ffn_layer_norm_bias;
    } layers[NUM_HIDDEN_LAYERS];

    float* pooler_weight;
    float* pooler_bias;
} Model;

// Function declarations
Tokenizer* load_tokenizer(const char* vocab_file);
Model* load_model(const char* model_file);
int* tokenize(Tokenizer* tokenizer, const char* text, int* num_tokens);
float* embed_text(const char* text, const char* vocab_file, const char* model_file);
void free_tokenizer(Tokenizer* tokenizer);
void free_model(Model* model);

#endif // EMBEDDING_MODEL_H
