#include <math.h>
#include "util.h"

float euclidean_distance(float* a, float* b, int dimensions) {
    float sum = 0.0f;
    for (int i = 0; i < dimensions; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

