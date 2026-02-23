#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define QUEUE_CAPACITY 1024
#define QUEUE_MASK (QUEUE_CAPACITY - 1)

// lock-free queue
template <typename T>
class LockFreeQueue {
    private:
        T buffer[QUEUE_CAPACITY];
        // Using unsigned int / uint32_t — atomicAdd works best with these
        unsigned int d_head;
        unsigned int d_tail;
        unsigned int capacity;

    public:
        // Constructor: Initialize head and tail to 0
        __host__ LockFreeQueue(): capacity(QUEUE_CAPACITY) {
            cudaMalloc(&d_head, sizeof(unsigned int));
            cudaMalloc(&d_tail, sizeof(unsigned int));
            cudaMemset(d_head, 0, sizeof(unsigned int));
            cudaMemset(d_tail, 0, sizeof(unsigned int));
        }

        // Destructor: Free device memory
        __host__ ~LockFreeQueue() {
            cudaFree(d_head);
            cudaFree(d_tail);
        }

        // Enqueue: Add an item to the queue
        __device__ bool push(const T& item) {
            unsigned int head = atomicAdd(d_head, 1);

            // full = (head + 1) % capacity == tail (leave one slot empty to distinguish full vs empty)
            unsigned int next_head = (head + 1) & QUEUE_MASK;
            unsigned int tail = *d_tail; // read tail atomically

            if (next_head == tail) {
                // Queue is full, revert head increment
                atomicSub(d_head, 1);
                return false; // push failed
            }

            buffer[head & QUEUE_MASK] = item; // write item to buffer
            return true; // push succeeded
        }

        // Dequeue: Remove an item from the queue
        __device__ bool pop(T& item) {
            unsigned int tail = atomicAdd(d_tail, 1);

            // empty = head == tail
            unsigned int head = *d_head; // read head atomically

            if (tail == head) {
                // Queue is empty, revert tail increment
                atomicSub(d_tail, 1);
                return false; // pop failed
            }

            item = buffer[tail & QUEUE_MASK]; // read item from buffer
            return true; // pop succeeded
        }

        __device__ unsigned int size_approx() const {
        unsigned int h = *d_head;
        unsigned int t = *d_tail;
        return (h - t) & 0x7FFFFFFF;
    }
};

// producer kernel: pushes items into the queue
__global__ void producer(LockFreeQueue<int>* q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    q->push(idx);
}

// consumer kernel: pops items from the queue
__global__ void consumer(LockFreeQueue<int>* q, int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value;
    if (q->pop(value)) {
        output[idx] = value; // store popped value in output array
    } else {
        output[idx] = -1; // indicate pop failure (queue empty)
    }
}

// host function to test the lock-free queue
int main() {
    LockFreeQueue<int>* d_queue;
    cudaMalloc(&d_queue, sizeof(LockFreeQueue<int>));
    LockFreeQueue<int> h_queue;
    cudaMemcpy(d_queue, &h_queue, sizeof(LockFreeQueue<int>), cudaMemcpyHostToDevice);

    // allocate output buffer for consumer results
    int *d_output;
    int* h_output = new int[32];
    cudaMalloc(&d_output, 32 * sizeof(int));

    // Launch producer kernel (push 32 items)
    producer<<<1, 32>>>(d_queue);
    cudaDeviceSynchronize();

    // Launch consumer kernel (pop 32 items)
    consumer<<<1, 32>>>(d_queue, d_output);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Queue results:\n");
    for (int i = 0; i < 32; i++) {
        printf("Thread %d popped value: %d\n", i, h_output[i]);
    }

    // Cleanup
    cudaFree(d_queue);
    cudaFree(d_output);
    delete[] h_output;

    return 0;

}