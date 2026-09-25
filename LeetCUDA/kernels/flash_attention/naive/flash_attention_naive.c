#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

/*
Standard Implementation of Self-Attention in pure C.
 Q, K, V : (N x d) input matrices
   O       : (N x d) output matrix
   N       : sequence length
   d       : head dimension
   causal  : if non-zero, apply a causal mask (position i can only
             attend to positions j <= i) -- standard for autoregressive
             decoder self-attention.

 Implements:
   S = (Q K^T) * (1/sqrt(d))
   P = row-wise softmax(S)      (masked entries -> 0 probability)
   O = P V
*/

// Allocate a NxN matrix of floats, stored in row-major as flat-array.
static float *mat_alloc(int n, int m) {
    float *p = (float *)malloc((size_t)n * (size_t)m * sizeof(float));
    if (!p) {
        fprintf(stderr, "malloc failed for %dx%d matrix\n", n, m);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void mat_free(float *m) { free(m); }

static void mat_fill_random(float *m, int n, int cols, float scale) {
    for (int i = 0; i < n * cols; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
        m[i] = r * scale;
    }
}
 
static void mat_print(const char *name, const float *m, int n, int cols) {
    printf("%s (%dx%d):\n", name, n, cols);
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("% .4f", m[i * cols + j]);
            if (j + 1 < cols) printf(", ");
        }
        printf("]\n");
    }
}
static void softmax(float *raw_scores, int len) {
    float max_val = raw_scores[0];

    for (int i = 1; i < len; i++) {
        if (raw_scores[i] > max_val) max_val = raw_scores[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        raw_scores[i] = expf(raw_scores[i] - max_val);
        sum += raw_scores[i];
    }

    for (int i = 0; i < len; i++) {
        raw_scores[i] /= sum;
    }
}

static void self_attention(const float *Q, const float *K, const float *V, float *O, int seq_l, int d) {
    float scale = (1.0 / sqrt((float)d));

    for (int q = 0; q < seq_l; q++) {
        float s[seq_l];
        for (int i = 0; i < seq_l; i++) {
            s[i] = 0.0f;
            for (int j = 0; j < d; j++) {
                s[i] += Q[q * d + j] * K[i * d + j];
            }
            s[i] *= scale;
        }

        // Convert the raw scores into attention weights
        softmax(s, seq_l);

        // output
        for (int i = 0; i < d; i++) {
            O[q * d + i] = 0.0f;
            for (int j = 0; j < seq_l; j++) {
                O[q * d + i] += s[j] * V[j * d + i];
           }
        }
    }
}

int main(void) {
    srand(42);

    int seq_l = 6;
    int d = 4;

    float *Q = mat_alloc(seq_l, d);
    float *K = mat_alloc(seq_l, d);
    float *V = mat_alloc(seq_l, d);
    float *O = mat_alloc(seq_l, d);

    mat_fill_random(Q, seq_l, d, 1.0);
    mat_fill_random(K, seq_l, d, 1.0);
    mat_fill_random(V, seq_l, d, 1.0);

    printf("Self-attention demo: N=%d (seq len), d=%d (head dim)", seq_l, d);
    mat_print("Q", Q, seq_l, d);
    mat_print("K", K, seq_l, d);
    mat_print("V", V, seq_l, d);

    self_attention(Q, K, V, O, seq_l, d);
 
    printf("\n");
    mat_print("O", O, seq_l, d);

    printf("OUTPUT O = (");
    for (int i = 0; i < d - 1; i++) {
        printf("%.4f, ", O[i]);
    }
    printf("%.4f)\n", O[d - 1]);
 
    mat_free(Q);
    mat_free(K);
    mat_free(V);
    mat_free(O);
 
    return 0;
}