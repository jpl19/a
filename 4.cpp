gcc prg.cpp -o prg
./prg

nvcc -o yy_program yy.cu
  ./yy_program

//bubble
#include <iostream>
#include <vector>
#include <omp.h>
#include<ctime>

using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped = true;

    while (swapped) {
        swapped = false;
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    }
}

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp;
    int left = l;
    int right = m + 1;

    while (left <= m && right <= r) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            right++;
        }
    }

    while (left <= m) {
        temp.push_back(arr[left]);
        left++;
    }

    while (right <= r) {
        temp.push_back(arr[right]);
        right++;
    }

    for (int i = l; i <= r; i++) {
        arr[i] = temp[i - l];
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);
            #pragma omp section
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    clock_t bubbleStart = clock();
    bubbleSort(arr);
    clock_t bubbleEnd = clock();
    cout << "Sorted array using Bubble Sort: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    clock_t mergeStart = clock();
    mergeSort(arr, 0, n - 1);
    clock_t mergeEnd = clock();
    cout << "Sorted array using Merge Sort: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    double bubbleDuration = double(bubbleEnd-bubbleStart) / CLOCKS_PER_SEC;
    double mergeDuration = double(mergeEnd-mergeStart) / CLOCKS_PER_SEC;

    cout << "Bubble sort time in seconds: " << bubbleDuration << endl;
    cout << "Merge sort time in seconds: " << mergeDuration << endl;

    return 0;
}
//cuda 
#include <stdio.h>
#include <stdlib.h>

__global__
void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

__global__
void multiply(int* A, int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initializeVector(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;
    }
}

void printVector(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

void printMatrix(int* matrix, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            printf("%d ", matrix[row * size + col]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int N = 4;

    // Vector addition
    int* A, * B, * C;
    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);

    A = (int*)malloc(vectorBytes);
    B = (int*)malloc(vectorBytes);
    C = (int*)malloc(vectorBytes);

    initializeVector(A, vectorSize);
    initializeVector(B, vectorSize);

    printf("Vector A: ");
    printVector(A, N);
    printf("Vector B: ");
    printVector(B, N);

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlockVec = 256;
    int blocksPerGridVec = (N + threadsPerBlockVec - 1) / threadsPerBlockVec;

    add<<<blocksPerGridVec, threadsPerBlockVec>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    printf("Addition: ");
    printVector(C, N);

    free(A);
    free(B);
    free(C);

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    // Matrix multiplication
    int* D, * E, * F;
    int matrixSize = N;
    size_t matrixBytes = matrixSize * matrixSize * sizeof(int);

    D = (int*)malloc(matrixBytes);
    E = (int*)malloc(matrixBytes);
    F = (int*)malloc(matrixBytes);

    initializeMatrix(D, matrixSize);
    initializeMatrix(E, matrixSize);

    printf("\nMatrix D: \n");
    printMatrix(D, matrixSize);

    printf("Matrix E: \n");
    printMatrix(E, matrixSize);

    int* M, * NMat, * O;
    cudaMalloc(&M, matrixBytes);
    cudaMalloc(&NMat, matrixBytes);
    cudaMalloc(&O, matrixBytes);

    cudaMemcpy(M, D, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(NMat, E, matrixBytes, cudaMemcpyHostToDevice);

    int threadsPerBlockMat = 2;
    int blocksPerGridMat = (matrixSize + threadsPerBlockMat - 1) / threadsPerBlockMat;

    dim3 threadsMat(threadsPerBlockMat, threadsPerBlockMat);
    dim3 blocksMat(blocksPerGridMat, blocksPerGridMat);

    multiply<<<blocksMat, threadsMat>>>(M, NMat, O, matrixSize);

    cudaMemcpy(F, O, matrixBytes, cudaMemcpyDeviceToHost);
    printf("Multiplication: \n");
    printMatrix(F, matrixSize);

    free(D);
    free(E);
    free(F);

    cudaFree(M);
    cudaFree(NMat);
    cudaFree(O);

    return 0;
}
