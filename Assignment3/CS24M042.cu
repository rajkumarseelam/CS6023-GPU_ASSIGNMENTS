#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#define MOD 1000000007
#define BLOCK_SIZE 1024

using std::cin;
using std::cout;
using std::vector;
using std::string;



//without path compression 
__device__ int find1(int *device_parent, int i) {
    while (device_parent[i] != i) {
        i = device_parent[i];
    }
    return i;
}

//Using these two combinely

//with path compression
__device__ int find2(int *device_parent, int i) {
    int root = i;
    while (device_parent[root] != root) {
        root = device_parent[root];
    }

    while (i != root) {
        int next = device_parent[i];
        device_parent[i] = root;
        i = next;
    }

    return root;
}



__global__ void Component_Initialize(int *device_parent, int V){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V){
    device_parent[tid] = tid;
    }
}
__global__ void weight_computation( int E, int *device_weight_conversion,int *device_weight){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E){
        device_weight[tid]*=device_weight_conversion[tid];
    }
}


__global__ void reset_min_comp_weight(int *device_min_comp_weight, int V, int E,int *device_min_comp_edge) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        device_min_comp_weight[tid] = MOD;
        device_min_comp_edge[tid]=MOD;
    }
}


__device__ void union_check(int edge,int w,unsigned long long *result,int *device_src,int *device_dest,int *device_parent, int *device_rank,int *device_total_components) {
        int u = device_src[edge];
        int v = device_dest[edge];    

        int set1 = find1(device_parent, u);
        int set2 = find1(device_parent, v);

        while (1) {
            int n1 = find1(device_parent, u);
            int n2 = find1(device_parent, v);
            
            if (n1 == n2) break;  
            
            //swapping to remove the circular loop
            if (n1>n2) {
                long int temp =n1;
                n1= n2;
                n2 = temp;
            }
    
            if (device_parent[n1] != n1) continue;
            if (atomicCAS((int *)&device_parent[n1], n1, n2) == n1) {
                atomicAdd(result, w);
                atomicAdd(device_total_components, -1);
                break;
            }
        }

}


__global__ void perform_union_components(
    int V, 
    int *device_parent, int *device_rank, int* device_min_comp_edge, 
    unsigned long long *result, int *device_total_components,int *device_src ,int *device_dest,int *device_weight,int E)
{

     __shared__ unsigned long long localweight;
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    localweight=0;
    __syncthreads();


    if (node < V){
        int tid = device_min_comp_edge[node];

        if(tid != MOD) {
            int w=device_weight[tid];

            int edge=tid;

            union_check(edge,w,&localweight,device_src,device_dest,device_parent,device_rank,device_total_components);
        }
    }
    __syncthreads();

    if(threadIdx.x ==0){
        atomicAdd(result,localweight);
    }


} 


__global__ void find_min_weight_components(
    int *device_src, int *device_dest, int *device_weight, 
    int *device_parent, int *device_rank, int *device_min_comp_weight,
    unsigned long long *result, int V, int E, int *device_store_parent)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E) {
        int u = device_src[tid];
        int v = device_dest[tid];
        int w = device_weight[tid];
        int first=device_store_parent[u];
        int second= device_store_parent[v];
        if (first != second) {
            
            atomicMin(&device_min_comp_weight[first],w);
            atomicMin(&device_min_comp_weight[second],w);

        }
    }
}

__global__ void find_min_weight_edges(int *device_store_parent, int *device_src, int *device_dest ,int *device_min_comp_weight,int E,int* device_min_comp_edge,int *device_weight){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<E){

        int n1 = device_src[tid];
        int n2 = device_dest[tid];

        int u=device_store_parent[n1];
        int v=device_store_parent[n2];
        int w = device_weight[tid];
        if(u !=v){
            if(device_min_comp_weight[u]==w){
                atomicCAS(&device_min_comp_edge[u],MOD,tid);
            }
            if(device_min_comp_weight[v]==w){
                atomicCAS(&device_min_comp_edge[v],MOD,tid);
            }
        }
    }
}


__global__ void parent_store(int *device_store_parent, int *device_parent, int V) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        device_store_parent[tid] = find2(device_parent, device_parent[tid]);
    }
}

__global__ void find_MOD(unsigned long long *result){
    *result=*result%MOD;
}

int main() {

    int V, E;
    cin >> V >> E;
    
    int *src = new int[E];
    int *dest = new int[E];
    int *weight = new int[E];
    int *weight_conversion = new int[E];
    for (int i = 0; i < E; ++i) {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt >> s;
    
        src[i] = u;
        dest[i] = v;
        weight[i] = wt;
    
        if (s == "green")
            weight_conversion[i] = 2;
        else if (s == "traffic")
            weight_conversion[i] = 5;
        else if (s == "dept")
            weight_conversion[i] = 3;
        else
            weight_conversion[i] = 1;
    }

    int *device_src, *device_dest, *device_weight;
    int *device_parent;
    int *device_rank, *device_min_comp_weight, *device_min_comp_edge;
    unsigned long long *device_total_weight;
    int *device_total_components;
    int *device_store_parent;
    int *device_weight_conversion;

    cudaMalloc(&device_src, E * sizeof(int));
    cudaMalloc(&device_dest, E * sizeof(int));
    cudaMalloc(&device_weight, E * sizeof(int));
    cudaMalloc(&device_weight_conversion, E * sizeof(int));
    cudaMalloc(&device_parent, V * sizeof(int));
    cudaMalloc(&device_store_parent, V * sizeof(int));
    cudaMalloc(&device_rank, V * sizeof(int));
    cudaMalloc(&device_min_comp_weight, V * sizeof(int));
    cudaMalloc(&device_min_comp_edge, V * sizeof(int));
    cudaMalloc(&device_total_weight, sizeof(unsigned long long));
    cudaMalloc(&device_total_components, sizeof(int));


    cudaMemcpy(device_src, src, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dest, dest, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight_conversion, weight_conversion, E * sizeof(int), cudaMemcpyHostToDevice);


    unsigned long long host_total_weight = 0;
    cudaMemcpy(device_total_weight, &host_total_weight, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    int host_total_components = V;
    cudaMemcpy(device_total_components, &host_total_components, sizeof(int), cudaMemcpyHostToDevice);

    
    int gridV = ceilf(float (V) / BLOCK_SIZE);
    int gridE = ceilf( float (E)/ BLOCK_SIZE);


    
    cudaMemset(device_rank, 0, sizeof(int));
    
    
    auto start = std::chrono::high_resolution_clock::now();
   
    
    //edge intialization
    Component_Initialize<<<gridV, BLOCK_SIZE>>>(device_parent, V);
    //weight conversion
    weight_computation<<<gridE,BLOCK_SIZE>>>(E, device_weight_conversion, device_weight);
    

    // int iterations=1;

    while (host_total_components > 1) {
        cudaMemcpy(device_total_components, &host_total_components, sizeof(int), cudaMemcpyHostToDevice);
        // cout << "Iteration " << iterations << " start\n";
        reset_min_comp_weight<<<gridV, BLOCK_SIZE>>>(device_min_comp_weight, V,E,device_min_comp_edge);

        parent_store<<<gridV, BLOCK_SIZE>>>(device_store_parent,device_parent,V);

        find_min_weight_components<<<gridE, BLOCK_SIZE>>>(device_src, device_dest, device_weight, device_parent, device_rank, device_min_comp_weight, device_total_weight, V, E, device_store_parent);

     

        find_min_weight_edges<<<gridE, BLOCK_SIZE>>>(device_store_parent, device_src, device_dest ,device_min_comp_weight,E,device_min_comp_edge,device_weight);
   
        perform_union_components<<<gridV, BLOCK_SIZE>>>(V, device_parent, device_rank, device_min_comp_edge, device_total_weight, device_total_components,device_src, device_dest,device_weight,E);

        
        cudaMemcpy(&host_total_components, device_total_components, sizeof(int), cudaMemcpyDeviceToHost);

        // cout << "Components = " << host_total_components << "\n";
        // iterations++;
        
    }

    
    find_MOD<<<1,1>>>(device_total_weight);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    cudaMemcpy(&host_total_weight, device_total_weight, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    // cout <<"time = " <<elapsed.count() << " s\n";
    cout << host_total_weight<< "\n";

    // Cleanup.
    delete[] src;
    delete[] dest;
    delete[] weight;
    delete[] weight_conversion;
    cudaFree(device_src);
    cudaFree(device_dest);
    cudaFree(device_weight);
    cudaFree(device_parent);
    cudaFree(device_rank);
    cudaFree(device_min_comp_weight);
    cudaFree(device_total_weight);
    cudaFree(device_total_components);
    cudaFree(device_store_parent);
    cudaFree(device_min_comp_edge);
    cudaFree(device_weight_conversion);

    cudaDeviceSynchronize();
    // std::ofstream file("cuda.out");
    // if (file.is_open())
    // {
    //     file << host_total_weight << "\n";
    //     file.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    // std::ofstream file2("cuda_timing.out");
    // if (file2.is_open())
    // {
    //     file2 << elapsed.count() << "\n";
    //     file2.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    return 0;
}