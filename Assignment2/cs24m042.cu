#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//replaced cuda/cuda_runtime.h with cuda_runtime.h

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;
//created constant memory for to store the filter
__constant__ long int filter[4096]; 

__global__ void dkernel(long int *matrix, long int *result, int h, int w, int c, int r, int s, int k)
{
    // sample kernel you can use your own kernel

    extern __shared__  long int sharedMemory[];  

    //created Dynamic shared memory which stores one row of the input data i.e each thread is working on one row,column value of the input. 
    long int* row = sharedMemory; 
    //Storing the computed values in output array, using this I was leveraging  the use of shared memory
    long int* output = &sharedMemory[w];  
   
    long int index=blockIdx.y * h* w + blockIdx.z * w +threadIdx.x;
    long int filter_index=blockIdx.x*c*r*s + blockIdx.y *r*s;
    long int result_index=blockIdx.x * h*w +blockIdx.z*w;

    row[threadIdx.x]=matrix[index];
    __syncthreads();

    int row_index=r/2,column_index=s/2;

    int mr=(s+1)/2 ,ml=s-mr;

    int t=threadIdx.x;
    int cur_block=blockIdx.z;

    //For a input block I'm sliding the filter array top to down.. Intially I have splitted into two parts mid to low and mid+1 to high .. Later optimised the code by combining the two for loops.
    
    int mtolow_sz=min(row_index+1,h-cur_block);
    int mtohigh_sz=min(row_index,cur_block);

    for(int i=row_index-mtolow_sz+1;i<=row_index+mtohigh_sz;i++){
        output[t]=0;
        // To get better coalescing and reduce the thread divergence , I tried to split the filter array to two parts and slided on the input block(row). 
        int sum1=0,start,last;
        start=filter_index + i*s +column_index;
        last=min(w,mr);

        //Sliding the right half of filter array.
        for(int m=0;m<last;m++){
            if(t+m<w){
                sum1+=row[t+m]*filter[start+m];
            }
        }
        
        output[t]=sum1;
        __syncthreads();

        int sum2=0;
        start=filter_index + i*s + column_index-1;
        last=min(w-1,ml);
        
        //sliding the left half of the filter array.
        for(int m=0;m<last;m++){
            if(t-m>=0){
                sum2+=(row[t-m]*filter[start-m]);
            }   
        }

        if(t!=w-1){
        output[t+1]+=sum2;
        }
        __syncthreads();

        //Adding the value to the corresponding result row,column pair.
        int ri=result_index + (row_index-i)*w;
        atomicAdd((unsigned long long int *)&result[ri +t], (unsigned long long int)output[t]);

    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    long int *result;
    long int *matrix;

    //Creating memory for storing result and matrix.
    cudaMalloc(&matrix,h*w*c* sizeof(long int));
    cudaMalloc(&result, h*w*k* sizeof(long int));
    

    cudaMemcpy(matrix, h_mat, h*w*c* sizeof(long int), cudaMemcpyHostToDevice);
    //Copied the filter into constant memory.
    cudaMemcpyToSymbol(filter, h_filter, r * s * c * k * sizeof(long int));

    //setting intial result values to 0.
    cudaMemset(result, 0, h*w*k* sizeof(long int));
    

    dim3 grid_sz(k,c,h);
    dim3 block_sz(w,1,1);

    size_t sharedMemorySize = 2 * w * sizeof(long int);

    dkernel<<<grid_sz,block_sz,sharedMemorySize>>>(matrix,result,h,w,c,r,s,k);
    
    cudaMemcpy(h_ans, result, h*w*k* sizeof(long int), cudaMemcpyDeviceToHost);

    //Free the assigned memory from the GPU.
    cudaFree(matrix);
    cudaFree(result);



    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
