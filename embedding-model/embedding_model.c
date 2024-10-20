#include "embedding_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EMBEDDING_DIM 384  // Update this to match your model's embedding dimension
#define VOCAB_SIZE 30522  // Update this to match your model's vocabulary size
#define MAX_SEQ_LENGTH 512  // Update this if your model uses a different max sequence length
#define LAYER_NORM_EPS 1e-12f

static Tokenizer* tokenizer = NULL;
static Model* model = NULL;

Tokenizer* load_tokenizer(const char* vocab_file) {
    Tokenizer* t = malloc(sizeof(Tokenizer));
    t->vocab = malloc(VOCAB_SIZE * sizeof(char*));
    t->vocab_size = VOCAB_SIZE;

    FILE* file = fopen(vocab_file, "r");
    if (!file) {
        fprintf(stderr, "Failed to open vocab file\n");
        return NULL;
    }

    char line[256];
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < VOCAB_SIZE) {
        line[strcspn(line, "\n")] = 0;  // Remove newline
        t->vocab[i] = strdup(line);
        i++;
    }

    fclose(file);
    return t;
}

Model* load_model(const char* model_file) {
    Model* m = malloc(sizeof(Model));
    
    FILE* file = fopen(model_file, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open model file\n");
        return NULL;
    }

    // Read embeddings
    m->token_embeddings = malloc(VOCAB_SIZE * EMBEDDING_DIM * sizeof(float));
    fread(m->token_embeddings, sizeof(float), VOCAB_SIZE * EMBEDDING_DIM, file);

    m->position_embeddings = malloc(MAX_SEQ_LENGTH * EMBEDDING_DIM * sizeof(float));
    fread(m->position_embeddings, sizeof(float), MAX_SEQ_LENGTH * EMBEDDING_DIM, file);

    m->token_type_embeddings = malloc(2 * EMBEDDING_DIM * sizeof(float));
    fread(m->token_type_embeddings, sizeof(float), 2 * EMBEDDING_DIM, file);

    m->embeddings_layer_norm_weight = malloc(EMBEDDING_DIM * sizeof(float));
    fread(m->embeddings_layer_norm_weight, sizeof(float), EMBEDDING_DIM, file);

    m->embeddings_layer_norm_bias = malloc(EMBEDDING_DIM * sizeof(float));
    fread(m->embeddings_layer_norm_bias, sizeof(float), EMBEDDING_DIM, file);

    // Read layers
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        // Attention weights
        m->layers[i].attention.query = malloc(EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention.query, sizeof(float), EMBEDDING_DIM * EMBEDDING_DIM, file);

        m->layers[i].attention.key = malloc(EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention.key, sizeof(float), EMBEDDING_DIM * EMBEDDING_DIM, file);

        m->layers[i].attention.value = malloc(EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention.value, sizeof(float), EMBEDDING_DIM * EMBEDDING_DIM, file);

        m->layers[i].attention.output = malloc(EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention.output, sizeof(float), EMBEDDING_DIM * EMBEDDING_DIM, file);

        // Attention layer norm
        m->layers[i].attention_layer_norm_weight = malloc(EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention_layer_norm_weight, sizeof(float), EMBEDDING_DIM, file);

        m->layers[i].attention_layer_norm_bias = malloc(EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].attention_layer_norm_bias, sizeof(float), EMBEDDING_DIM, file);

        // FFN weights
        m->layers[i].ffn.intermediate = malloc(EMBEDDING_DIM * INTERMEDIATE_SIZE * sizeof(float));
        fread(m->layers[i].ffn.intermediate, sizeof(float), EMBEDDING_DIM * INTERMEDIATE_SIZE, file);

        m->layers[i].ffn.output = malloc(INTERMEDIATE_SIZE * EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].ffn.output, sizeof(float), INTERMEDIATE_SIZE * EMBEDDING_DIM, file);

        // FFN layer norm
        m->layers[i].ffn_layer_norm_weight = malloc(EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].ffn_layer_norm_weight, sizeof(float), EMBEDDING_DIM, file);

        m->layers[i].ffn_layer_norm_bias = malloc(EMBEDDING_DIM * sizeof(float));
        fread(m->layers[i].ffn_layer_norm_bias, sizeof(float), EMBEDDING_DIM, file);
    }

    // Pooler
    m->pooler_weight = malloc(EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float));
    fread(m->pooler_weight, sizeof(float), EMBEDDING_DIM * EMBEDDING_DIM, file);

    m->pooler_bias = malloc(EMBEDDING_DIM * sizeof(float));
    fread(m->pooler_bias, sizeof(float), EMBEDDING_DIM, file);

    fclose(file);
    return m;
}

int* tokenize(Tokenizer* tokenizer, const char* text, int* num_tokens) {
    int* tokens = malloc(MAX_SEQ_LENGTH * sizeof(int));
    *num_tokens = 0;

    char* text_copy = strdup(text);
    char* token = strtok(text_copy, " ");
    while (token != NULL && *num_tokens < MAX_SEQ_LENGTH) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            if (strcmp(token, tokenizer->vocab[i]) == 0) {
                tokens[(*num_tokens)++] = i;
                break;
            }
        }
        token = strtok(NULL, " ");
    }

    free(text_copy);
    return tokens;
}

void layer_norm(float* input, float* output, float* weight, float* bias, int size) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;

    float std = sqrt(var + LAYER_NORM_EPS);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / std * weight[i] + bias[i];
    }
}

void gelu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = 0.5f * input[i] * (1.0f + tanhf(0.797884f * (input[i] + 0.044715f * input[i] * input[i] * input[i])));
    }
}

void matrix_multiply(float* a, float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                c[i * k + j] += a[i * n + l] * b[l * k + j];
            }
        }
    }
}

float* embed_text(const char* text) {
    if (!tokenizer) tokenizer = load_tokenizer("vocab.txt");
    if (!model) model = load_model("model.bin");

    int num_tokens;
    int* tokens = tokenize(tokenizer, text, &num_tokens);

    float* embedding = calloc(EMBEDDING_DIM, sizeof(float));
    float* layer_input = calloc(EMBEDDING_DIM, sizeof(float));
    float* layer_output = calloc(EMBEDDING_DIM, sizeof(float));
    float* attention_output = calloc(EMBEDDING_DIM, sizeof(float));
    float* ffn_intermediate = calloc(INTERMEDIATE_SIZE, sizeof(float));

    // Embedding layer
    for (int i = 0; i < num_tokens; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding[j] += model->token_embeddings[tokens[i] * EMBEDDING_DIM + j];
            embedding[j] += model->position_embeddings[i * EMBEDDING_DIM + j];
            embedding[j] += model->token_type_embeddings[0 * EMBEDDING_DIM + j];  // Assuming token_type_id = 0
        }
    }

    layer_norm(embedding, layer_input, model->embeddings_layer_norm_weight, model->embeddings_layer_norm_bias, EMBEDDING_DIM);

    // Transformer layers
    for (int layer = 0; layer < NUM_HIDDEN_LAYERS; layer++) {
        // Self-attention
        matrix_multiply(layer_input, model->layers[layer].attention.query, attention_output, 1, EMBEDDING_DIM, EMBEDDING_DIM);
        matrix_multiply(layer_input, model->layers[layer].attention.key, layer_output, 1, EMBEDDING_DIM, EMBEDDING_DIM);
        matrix_multiply(layer_input, model->layers[layer].attention.value, embedding, 1, EMBEDDING_DIM, EMBEDDING_DIM);

        // Simplified attention calculation (this should be more complex in a full implementation)
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            attention_output[i] *= layer_output[i];
            attention_output[i] *= embedding[i];
        }

        matrix_multiply(attention_output, model->layers[layer].attention.output, layer_output, 1, EMBEDDING_DIM, EMBEDDING_DIM);

        // Add & Norm
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            layer_output[i] += layer_input[i];
        }
        layer_norm(layer_output, attention_output, model->layers[layer].attention_layer_norm_weight, model->layers[layer].attention_layer_norm_bias, EMBEDDING_DIM);

        // Feed-forward network
        matrix_multiply(attention_output, model->layers[layer].ffn.intermediate, ffn_intermediate, 1, EMBEDDING_DIM, INTERMEDIATE_SIZE);
        gelu(ffn_intermediate, INTERMEDIATE_SIZE);
        matrix_multiply(ffn_intermediate, model->layers[layer].ffn.output, layer_output, 1, INTERMEDIATE_SIZE, EMBEDDING_DIM);

        // Add & Norm
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            layer_output[i] += attention_output[i];
        }
        layer_norm(layer_output, layer_input, model->layers[layer].ffn_layer_norm_weight, model->layers[layer].ffn_layer_norm_bias, EMBEDDING_DIM);
    }

    // Pooler (simplified, just using the first token)
    matrix_multiply(layer_input, model->pooler_weight, embedding, 1, EMBEDDING_DIM, EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] = tanhf(embedding[i] + model->pooler_bias[i]);
    }

    free(tokens);
    free(layer_input);
    free(layer_output);
    free(attention_output);
    free(ffn_intermediate);

    return embedding;
}

void free_tokenizer(Tokenizer* tokenizer) {
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer);
}

void free_model(Model* model) {
    free(model->token_embeddings);
    free(model->position_embeddings);
    free(model->token_type_embeddings);
    free(model->embeddings_layer_norm_weight);
    free(model->embeddings_layer_norm_bias);

    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        free(model->layers[i].attention.query);
        free(model->layers[i].attention.key);
        free(model->layers[i].attention.value);
        free(model->layers[i].attention.output);
        free(model->layers[i].attention_layer_norm_weight);
        free(model->layers[i].attention_layer_norm_bias);
        free(model->layers[i].ffn.intermediate);
        free(model->layers[i].ffn.output);
        free(model->layers[i].ffn_layer_norm_weight);
        free(model->layers[i].ffn_layer_norm_bias);
    }

    free(model->pooler_weight);
    free(model->pooler_bias);
    free(model);
}

int main() {
    const char* text = "Hello, world!";
    float* embedding = embed_text(text);
    
    printf("Embedding vector:\n");
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        printf("%f ", embedding[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // Print 8 numbers per line for readability
    }
    
    free(embedding);  // Don't forget to free the allocated memory
    return 0;
}
