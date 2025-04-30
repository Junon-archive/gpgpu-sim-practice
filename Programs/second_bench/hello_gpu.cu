#include <stdio.h>

__global__ void hello_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d\n", idx);
}

int main() {
    // 간단한 커널 실행
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize(); // 커널이 끝날 때까지 대기
    return 0;
}