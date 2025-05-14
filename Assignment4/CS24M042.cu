// %%writefile run2.cu
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

using namespace std;

static const long long INF = (long long)4e18;

#define estimated_pathsize 10000
#define estimated_drops 1000
#define MAX_VISITED_ARRAY_SIZE 100000

struct Edge {
    int destination, distance;
};

//structure for the evacuation object
struct Evacuate_obj {
    int path_size;
    int drops_size;
    int path[estimated_pathsize]; 
    long long drops[estimated_drops][3];
};


//Intializing
__device__ void initializeDijkstraArrays(
    long long *dist, 
    int *prevv, 
    bool *visited, 
    int N, 
    int source
) {
    for (int i = 0; i < N; i++) {
        dist[i] = INF;
        prevv[i] = -1;
        visited[i] = false;
    }
    dist[source] = 0;
}

//Find midistvertex
__device__ int findMinDistVertex(
    const long long *dist, 
    const bool *visited, 
    int N
) {
    int u = -1;
    long long minDist = INF;
    for (int j = 0; j < N; j++) {
        if (visited[j] == false && minDist >dist[j] ) {
            minDist = dist[j];
            u = j; 
        }
    }
    return u;
}

//updateadjacentvertices
__device__ void updateAdjacentVertices(
    long long *dist, 
    int *prevv, 
    int u, 
    int *d_edges_to, 
    int *d_edges_len, 
    int *d_edge_offsets, 
    int *d_edge_count
) {
    int edge_start = d_edge_offsets[u];
    int edge_end = edge_start + d_edge_count[u];
    
    int e = edge_start;
        while (e < edge_end) {
        int v   = d_edges_to[e];
        int len = d_edges_len[e];


        if (!(dist[v] <= dist[u] + len )) {
            int temp1 = dist[u];
            dist[v]  = temp1 + len;
            prevv[v] = u;
        }
    ++e;
    }

}

//find nearest unvisited shelter.
__device__ int findNearestUnvisitedShelter(
    int S, 
    bool *usedShel, 
    int *d_shelterCity, 
    int *d_sheltercapacity, 
    long long *dist, 
    long long &bestD, 
    int &bestCity
) {
    int bestIdx = -1;
    bestD = INF;
    bestCity = -1;  
    int si = 0;
    while (si < S) {
            if (usedShel[si]) {
                ++si;
                continue;
            }
            int c   = d_shelterCity[si];
            int cap = atomicAdd(&d_sheltercapacity[si], 0);
            if (cap > 0) {
                if (dist[c] < bestD) {
                    bestD    = dist[c];
                    bestCity = c;
                    bestIdx  = si;
                }
            }
            ++si;
    }
    return bestIdx;
}

__device__ int reconstructPathSegment(
    int *prevv, 
    int bestCity, 
    int cur, 
    int *seg
) {
    int seg_size = 0;
    int v = bestCity;
    while (v != -1 && v != cur) {
        seg[seg_size++] = v;
        v = prevv[v];
    }
    if (v == cur) {
        seg[seg_size++] = v;
    } else {
        return -1;
    }
    for (int lo = 0, hi = seg_size - 1; lo < hi; ++lo, --hi) {
        int tmp = seg[lo];
        seg[lo] = seg[hi];
        seg[hi] = tmp;
    }
    return seg_size;
}

__device__ void appendPathSegment(
    int *result_path, 
    int &result_path_size, 
    const int *seg, 
    int seg_size
) {
    for (int k = 1; k < seg_size; k++) {
        result_path[result_path_size++] = seg[k];
    }
}


//Find edge lenth
__device__ int findEdgeLength(
    int u, 
    int v, 
    int *d_edges_to, 
    int *d_edges_len, 
    int *d_edge_offsets, 
    int *d_edge_count
) {
    int start = d_edge_offsets[u];
    int cnt   = d_edge_count[u];
    int* to   = d_edges_to   + start;
    int* len  = d_edges_len  + start;

    int i = 0;
    while (i < cnt) {
        if (to[i] == v) {
            return len[i];
        }
        ++i;
    }
    return -1;
}



// Find the farthest point within elderly distance limit Used in evacuate_for_small
__device__ int calculateElderlyReachablePoint(
    const int *path, 
    int path_size, 
    int src, 
    int bestCity, 
    int maxDistElder, 
    int *d_edges_to, 
    int *d_edges_len, 
    int *d_edge_offsets, 
    int *d_edge_count
) {
    long long distFromSource = 0;
    int far = src;
    int idx = 0;

    while (idx + 1 < path_size) {
        int u = path[idx];
        int v = path[idx + 1];
        int w = findEdgeLength(u, v,d_edges_to,d_edges_len,d_edge_offsets,d_edge_count);

       
        if (w != -1) {
            distFromSource += w;

            if (distFromSource <= maxDistElder) {
                far = v;
            }

            if (v == bestCity) {
                break;
            }
        }

        ++idx;
    }

    return far;
}


//Used in evacuate_for_small

__device__ void placeElderlyAtShelter(
    long long &remE, 
    int bestIdx, 
    int bestCity, 
    int *d_sheltercapacity, 
    long long drops[][3], 
    int &drops_size
) {
    int oldCap, newCap, takeE;
    for (;;) {
        
        oldCap = atomicAdd(&d_sheltercapacity[bestIdx], 0);
        takeE = min(remE, (long long)oldCap);
        if (takeE == 0) {
            break;
        }
        newCap = oldCap - takeE;
    
        if (atomicCAS(&d_sheltercapacity[bestIdx], oldCap, newCap) == oldCap) {
            break;
        }

    }
      
    remE -= takeE;  
    if (takeE > 0) {
        drops[drops_size][0] = bestCity;
        drops[drops_size][1] = 0;
        drops[drops_size][2] = takeE;
        drops_size++;
    }   
    if (remE > 0) {
        drops[drops_size][0] = bestCity;
        drops[drops_size][1] = 0;
        drops[drops_size][2] = remE;
        drops_size++;
        remE = 0;
    }
}

__device__ void placePrimeAgeAtShelter(
    long long &remP, 
    int bestIdx, 
    int bestCity, 
    int *d_sheltercapacity, 
    long long drops[][3], 
    int &drops_size
) {
    
    int oldCap, newCap, takeP;
    for (;;) {
        oldCap = atomicAdd(&d_sheltercapacity[bestIdx], 0);
        takeP = min(remP, (long long)oldCap);
        if (takeP == 0) {
            break;
        }
        newCap = oldCap - takeP;
        if (atomicCAS(&d_sheltercapacity[bestIdx], oldCap, newCap) == oldCap) {
            break;
        }
    } 
    remP -= takeP;
    
    if (takeP > 0) {
        drops[drops_size][0] = bestCity;
        drops[drops_size][1] = takeP;
        drops[drops_size][2] = 0;
        drops_size++;
    }
}

__device__ void placeRemainingElderly(
    long long &remE, 
    const int *path, 
    int path_size, 
    int maxDistElder, 
    int *d_edges_to, 
    int *d_edges_len, 
    int *d_edge_offsets, 
    int *d_edge_count, 
    long long drops[][3], 
    int &drops_size
) {
    long long acc = 0;
    int far = path[0]; 
    {
        int k = 0;
        while (k + 1 < path_size) {
            int u = path[k];
            int v = path[k + 1];
            int w = findEdgeLength(u, v,d_edges_to,d_edges_len,d_edge_offsets,d_edge_count);
            if (w != -1) {
                acc += w;
                if (acc <= maxDistElder) {
                    far = v;
                } else {
                    break;
                }
            }
            ++k;
        }
    }
       
    drops[drops_size][0] = far;
    drops[drops_size][1] = 0;
    drops[drops_size][2] = remE;
    drops_size++;
    remE = 0;
}

__device__ void placeRemainingPrimeAge(
    long long &remP, 
    int cur, 
    long long drops[][3], 
    int &drops_size
) {
    drops[drops_size][0] = cur;
    drops[drops_size][1] = remP;
    drops[drops_size][2] = 0;
    drops_size++;
    remP = 0;
}

__global__ void evacuate_for_small(
    Evacuate_obj *d_results,
    int *d_cityToindex, int N, int S, int P,
    int *d_edges_to, int *d_edges_len, int *d_edge_offsets, int *d_edge_count,int *d_populatedPrime, int *d_populatedElder,
    int *d_shelterCity,
	int *d_sheltercapacity,
    long long *d_global_dist,
    int *d_global_prevv,
    bool *d_global_visited,int *d_populatedcity,int maxDistElder
) {
    int populatedcity_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (populatedcity_index < P){
    
     Evacuate_obj &result = d_results[populatedcity_index];
     result.drops_size = 0;
     result.path_size = 0;
    
    long long *dist = d_global_dist + populatedcity_index * N;
    int *prevv  = d_global_prevv + populatedcity_index * N;
    bool *visited = d_global_visited + populatedcity_index * N;
    
    int src = d_populatedcity[populatedcity_index];
    long long remP = d_populatedPrime[populatedcity_index];
    long long remE = d_populatedElder[populatedcity_index];
    int cur = src;

    result.path[result.path_size++] = cur;
 
    bool usedShel[estimated_drops] = {false};
    int usedCount = 0;
    
    while ((remP > 0 || remE > 0) && usedCount < S) {
        initializeDijkstraArrays(dist, prevv, visited, N, cur);

        for (int i = 0; i < N; i++) {
            int u = findMinDistVertex(dist, visited, N);
            
            if (u == -1 || dist[u] == INF) break;
            visited[u] = true;
            updateAdjacentVertices(dist, prevv, u, d_edges_to, d_edges_len, d_edge_offsets, d_edge_count);
        }
        long long bestD;
        int bestCity;
        int bestIdx = findNearestUnvisitedShelter(S, usedShel, d_shelterCity, d_sheltercapacity, dist, bestD, bestCity);
        
        if (bestIdx < 0) break;
        
        usedShel[bestIdx] = true;
        usedCount++;
        int seg[estimated_drops];
        int seg_size = reconstructPathSegment(prevv, bestCity, cur, seg);
        
        if (seg_size < 0) break;
    
        appendPathSegment(result.path, result.path_size, seg, seg_size);
        
        int elderFarthest = calculateElderlyReachablePoint(
            result.path, result.path_size, src, bestCity, maxDistElder,
            d_edges_to, d_edges_len, d_edge_offsets, d_edge_count
        );
        
        bool elderCan = (elderFarthest == bestCity);
        if (!elderCan && remE > 0) {
            auto& dropRow = result.drops[result.drops_size];
            dropRow[0] = elderFarthest;
            dropRow[1] = 0;
            dropRow[2] = remE;
            ++result.drops_size;
            remE = 0;
        }
         else if (elderCan && remE > 0) {
            placeElderlyAtShelter(remE, bestIdx, bestCity, d_sheltercapacity, result.drops, result.drops_size);
        }
        if (remP > 0) {
            placePrimeAgeAtShelter(remP, bestIdx, bestCity, d_sheltercapacity, result.drops, result.drops_size);
        }
        
        cur = bestCity;
    }
    if (remE > 0) {
        placeRemainingElderly(remE, result.path, result.path_size, maxDistElder,
            d_edges_to, d_edges_len, d_edge_offsets, d_edge_count, result.drops, result.drops_size);
    }
    if (remP > 0) {
        placeRemainingPrimeAge(remP, cur, result.drops, result.drops_size);
    }
}
}

__device__ void initEvacuationData(
    int src,
    Evacuate_obj &result,
    bool *visited, int N,
    int &cur, int &elderFarthest,
    long long &elderDist
) {
    result.path_size = 0;
    result.drops_size = 0;
    
    cur = src;
    elderFarthest = src;
    elderDist = 0;
    result.path[result.path_size++] = cur;
    for (int i = 0; i < N; i++) {
        visited[i] = false;
    }
    visited[cur] = true;
}
__device__ void processShelter(
    int cur, long long &remP, long long &remE,
    int *d_shelterCity, int *d_shelterCap,
    int S, long long elderDist, int maxDistElder,
    Evacuate_obj &result,
    bool &foundShelter
) {
    foundShelter = false;
    
    for (int si = 0; si < S; si++) {
        if (d_shelterCity[si] == cur) {
            int cap = atomicAdd(&d_shelterCap[si], 0);
            
            if (cap > 0) {
                if (remE > 0 && elderDist <= maxDistElder) {
                    int oldCap, newCap, takeE;
                    for (;;) {
                        oldCap = atomicAdd(&d_shelterCap[si], 0);
                        takeE = min(remE, (long long)oldCap);
                        if (takeE == 0) {
                            break;
                        }
                        newCap = oldCap - takeE;
                        if (atomicCAS(&d_shelterCap[si], oldCap, newCap) == oldCap) {
                            break;
                        }
                    }    
                    if (takeE > 0) {
                        auto &dropRow = result.drops[result.drops_size];
                        dropRow[0] = cur;
                        dropRow[1] = 0;
                        dropRow[2] = takeE;
                        ++result.drops_size;
                        remE -= takeE;
                    }
                    
                }
                if (remP > 0) {
                    int oldCap, newCap, takeP;
                    for (;;) {
                        oldCap = atomicAdd(&d_shelterCap[si], 0);
                        takeP = min(remP, (long long)oldCap);
                        if (takeP == 0) {
                            break;
                        }
                        newCap = oldCap - takeP;
                        if (atomicCAS(&d_shelterCap[si], oldCap, newCap) == oldCap) {
                            break;
                        }
                    }
                    
                    
                    if (takeP > 0) {
                        auto &dropRow = result.drops[result.drops_size];
                        dropRow[0] = cur;
                        dropRow[1] = takeP;
                        dropRow[2] = 0;
                        ++result.drops_size;
                        remP -= takeP;
                    }
                    
                }
                foundShelter = true;
                break;
            }
        }
    }
}

__device__ int findValidCities(
    int cur,
    bool *visited,
    int *d_edges_to, int *d_edge_offsets, int *d_edge_count,
    int *valid_cities
) {
    int edge_start = d_edge_offsets[cur];
    int edge_end = edge_start + d_edge_count[cur];
    int valid_count = 0;
    
    for (int e = edge_start; e < edge_end; e++) {
        int next_city = d_edges_to[e];
        if (!visited[next_city] && valid_count < 100) {
            valid_cities[valid_count++] = e;
        }
    }
    
    return valid_count;
}

__device__ bool moveRandomly(
    int &cur, bool *visited,
    int *d_edges_to, int *d_edges_len,
    int *valid_cities, int valid_count,
    curandState &local_state,
    long long &elderDist, int maxDistElder, int &elderFarthest,
    Evacuate_obj &result
) {
    if (valid_count <= 0) return false;
    int selected_idx = curand(&local_state) % valid_count;
    int edge_idx = valid_cities[selected_idx];
    
    int next_city = d_edges_to[edge_idx];
    int edge_len = d_edges_len[edge_idx];
    cur = next_city;
    visited[cur] = true;
    result.path[result.path_size++] = cur;
    elderDist += edge_len;
    if (elderDist <= maxDistElder) {
        elderFarthest = cur;
    }
    
    return true;
}

__device__ void placeRemainingPopulation(
    int cur, int elderFarthest,
    long long &remP, long long &remE,
    Evacuate_obj &result
) {
    if (remE > 0) {
        result.drops[result.drops_size][0] = elderFarthest;
        result.drops[result.drops_size][1] = 0;
        result.drops[result.drops_size][2] = remE;
        result.drops_size++;
        remE = 0;
    }
    if (remP > 0) {
     
        auto &dropRow = result.drops[result.drops_size];
        dropRow[0] =cur;
        dropRow[1] = remP;
        dropRow[2] = 0;
        ++result.drops_size;
        remP = 0;
    }
}

__global__ void evacuate_for_large(
    curandState *d_rng_states,  Evacuate_obj *d_results,bool *d_visited_arrays,
    int maxDistElder, int P,int S, int *d_populated_city,
    int N,int *d_edge_count, int *d_edge_offsets,
    int *d_edges_len,int *d_edges_to,int *d_shelterCap,
    int *d_shelterCity, int *d_populated_Elderly,
    int *d_populated_Prime
) {
    int pi = blockIdx.x * blockDim.x + threadIdx.x;
    if (pi < P){
    int src = d_populated_city[pi];
    long long remP = d_populated_Prime[pi], remE = d_populated_Elderly[pi];
    int cur;
    Evacuate_obj &result = d_results[pi];
    bool *visited = &d_visited_arrays[pi * N];
    curandState local_state = d_rng_states[pi];
    long long elderDist;
    int elderFarthest;
    initEvacuationData(src, result, visited, N, cur, elderFarthest, elderDist);
    while ((remP > 0 || remE > 0) && result.path_size < estimated_pathsize) {
        bool foundShelter;
        processShelter(cur, remP, remE, d_shelterCity, d_shelterCap, S, elderDist, maxDistElder, result, foundShelter);
        if (remP == 0 && remE == 0) break;
        int valid_cities[100];
        int valid_count = findValidCities(cur, visited, d_edges_to, d_edge_offsets, d_edge_count, valid_cities);
        if (!moveRandomly(cur, visited, d_edges_to, d_edges_len, valid_cities, valid_count,
                        local_state, elderDist, maxDistElder, elderFarthest, result)) {
            break;
        }
    }
    
    placeRemainingPopulation(cur, elderFarthest, remP, remE, result);
    d_rng_states[pi] = local_state;

    }
}

__global__ void initRNG(curandState *state, unsigned long long seed, int P) {
    int pi = blockIdx.x * blockDim.x + threadIdx.x;
    if (pi >= P) return;
    curand_init(seed, pi, 0, &state[pi]);
}



bool parseArgs(int argc, char** argv, ifstream& in, string& outFile) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <in> <out>\n";
        return false;
    }
    
    in.open(argv[1]);
    if (!in) {
        cerr << "Cannot open " << argv[1] << "\n";
        return false;
    }
    
    outFile = argv[2];
    return true;
}

// Read Input
bool readInputData(ifstream& in, int& Number_cities, int& maxDistElder,
                  vector<vector<Edge>>& adjacency_edge_list,
                  vector<int>& shelterCity, vector<int>& shelterCap, vector<int>& origCap,
                  unordered_map<int, int>& city_To_Id,
                  vector<int>& populated_city, vector<int>& populated_Prime, vector<int>& populated_Elderly) {
    int num_Roads;
    in >> Number_cities >> num_Roads;
    
    // Read graph
    adjacency_edge_list.resize(Number_cities);
    for (int i = 0; i < num_Roads; i++) {
        int u, v, len_Of_road, capacity_Of_road;
        in >> u >> v >> len_Of_road >> capacity_Of_road;
        adjacency_edge_list[v].push_back({u, len_Of_road});
        adjacency_edge_list[u].push_back({v, len_Of_road});
        
    }

    int Shelter_size;
    in >> Shelter_size;
    origCap.resize(Shelter_size);
    shelterCity.resize(Shelter_size);
    shelterCap.resize(Shelter_size);
    
    
    for (int i = 0; i < Shelter_size; i++) {
        in >> shelterCity[i] >> shelterCap[i];
        city_To_Id[shelterCity[i]] = i;
        origCap[i] = shelterCap[i];
        
    }

    int Pop_citysize;
    in >> Pop_citysize;
    populated_city.resize(Pop_citysize);
    populated_Prime.resize(Pop_citysize);
    populated_Elderly.resize(Pop_citysize);
    
    for (int i = 0; i < Pop_citysize; i++) {
        in >> populated_city[i] >> populated_Prime[i] >> populated_Elderly[i];
    }
    
    in >> maxDistElder;
    in.close();
    
    return true;
}


void covertAdjToCSR(const vector<vector<Edge>>& adjacency_edge_list, int N,
    vector<int>& edges_to, vector<int>& edges_len,
    vector<int>& edge_offsets, vector<int>& edge_count) {
    edges_to.clear();
    edges_len.clear();
    edge_offsets.clear();
    edge_count.clear();

    int runningOffset = 0;

    for (const auto& neighbors : adjacency_edge_list) {
    edge_offsets.push_back(runningOffset);
    edge_count.push_back(static_cast<int>(neighbors.size()));

    for (const auto& e : neighbors) {
    edges_to.push_back(e.destination);
    edges_len.push_back(e.distance);
    }

    runningOffset += static_cast<int>(neighbors.size());
    }
}



int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ifstream in;
    string outFile;
    if (!parseArgs(argc, argv, in, outFile)) {
        return 1;
    }
    
    // Read  parameters
    int N, maxDistElder;
    vector<vector<Edge>> adjacency_edge_list;
    vector<int> shelterCity, shelterCap, origCap;
    unordered_map<int, int> city_To_Id;
    vector<int> populated_city, populated_Prime, populated_Elderly;
    
    if (!readInputData(in, N, maxDistElder, adjacency_edge_list, shelterCity, shelterCap, origCap, 
                      city_To_Id, populated_city, populated_Prime, populated_Elderly)) {
        return 1;
    }
    
    int Pop_citysize = populated_city.size();
    int S = shelterCity.size();
    
    vector<int> edges_to, edges_len, edge_offsets, edge_count;
    covertAdjToCSR(adjacency_edge_list, N, edges_to, edges_len, edge_offsets, edge_count);

    vector<int> cityToid_vec;
    cityToid_vec.resize(N, -1);
    for (const auto& [city, idx] : city_To_Id) {
        cityToid_vec[city] = idx;
    }
    
    Evacuate_obj *d_results;
    int *d_edges_to, *d_edge_offsets, *d_edge_count;
    int *d_populated_city, *d_edges_len;
    int *d_shelterCity, *d_shelterCap;
    int *d_populated_Prime, *d_populated_Elderly;
    
    
    
    cudaMalloc(&d_populated_city, Pop_citysize * sizeof(int));
    cudaMalloc(&d_populated_Prime, Pop_citysize * sizeof(int));
    cudaMalloc(&d_populated_Elderly, Pop_citysize * sizeof(int));
    cudaMalloc(&d_edge_offsets, N * sizeof(int));
    cudaMalloc(&d_edge_count, N * sizeof(int));
    cudaMalloc(&d_shelterCity, S * sizeof(int));
    cudaMalloc(&d_shelterCap, S * sizeof(int));
    cudaMalloc(&d_edges_to, edges_to.size() * sizeof(int));
    cudaMalloc(&d_edges_len, edges_len.size() * sizeof(int));
    cudaMalloc(&d_results, Pop_citysize * sizeof(Evacuate_obj));
    
    //copy data to gpu
    cudaMemcpy(d_populated_city, populated_city.data(), Pop_citysize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_populated_Prime, populated_Prime.data(), Pop_citysize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_len, edges_len.data(), edges_len.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_offsets, edge_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_populated_Elderly, populated_Elderly.data(), Pop_citysize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelterCity, shelterCity.data(), S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelterCap, shelterCap.data(), S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_to, edges_to.data(), edges_to.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_count, edge_count.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int no_of_threads = 256;
    int no_of_blocks = (Pop_citysize + no_of_threads - 1) / no_of_threads;
    
    if (N <= 1000) {
        // using Dijkstra for this
        int *d_city_To_Id;
        long long *d_global_dist;
        int *d_global_prevv;
        bool *d_global_visited;
        
        vector<int> cityToid_vec(N, -1);
        for (const auto& [city, idx] : city_To_Id) {
            cityToid_vec[city] = idx;
        }
        
        cudaMalloc(&d_global_prevv, Pop_citysize * N * sizeof(int));
        cudaMalloc(&d_global_visited, Pop_citysize * N * sizeof(bool));
        cudaMalloc(&d_city_To_Id, N * sizeof(int));
        cudaMalloc(&d_global_dist, Pop_citysize * N * sizeof(long long));
        
        
        cudaMemcpy(d_city_To_Id, cityToid_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        
        evacuate_for_small<<<no_of_blocks, no_of_threads>>>(
            d_results, 
            d_city_To_Id, N, S, Pop_citysize, 
            d_edges_to, d_edges_len, d_edge_offsets, d_edge_count,d_populated_Prime, d_populated_Elderly,
            d_shelterCity,
			d_shelterCap,
            d_global_dist, d_global_prevv, d_global_visited,d_populated_city,maxDistElder
        );
        
        cudaFree(d_city_To_Id);
        cudaFree(d_global_dist);
        cudaFree(d_global_prevv);
        cudaFree(d_global_visited);
    } else {
        // Random move for large file i.e >1000
        curandState *d_rng_states;
        bool *d_visited_arrays;
        
        cudaMalloc(&d_rng_states, Pop_citysize * sizeof(curandState));
        cudaMalloc(&d_visited_arrays, Pop_citysize * N * sizeof(bool));
        
        // Initialize RNG states
        time_t seed = time(NULL);
        initRNG<<<no_of_blocks, no_of_threads>>>(d_rng_states, seed, Pop_citysize);
        
        evacuate_for_large<<<no_of_blocks, no_of_threads>>>(
            d_rng_states,  d_results,   d_visited_arrays,
            maxDistElder,Pop_citysize,S,d_populated_city,
            N,d_edge_count, d_edge_offsets,
            d_edges_len,d_edges_to,  d_shelterCap,
            d_shelterCity, d_populated_Elderly,
            d_populated_Prime 
        );
        
        cudaFree(d_rng_states);
        cudaFree(d_visited_arrays);
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back
    vector<Evacuate_obj> results(Pop_citysize);
    cudaMemcpy(results.data(), d_results, Pop_citysize * sizeof(Evacuate_obj), cudaMemcpyDeviceToHost);
    
    // Clean
    cudaFree(d_populated_city);
    cudaFree(d_populated_Prime);
    cudaFree(d_populated_Elderly);
    cudaFree(d_shelterCity);
    cudaFree(d_shelterCap);
    cudaFree(d_edges_to);
    cudaFree(d_edges_len);
    cudaFree(d_edge_offsets);
    cudaFree(d_edge_count);
    cudaFree(d_results);
    
    // declared and allocate memory for the required variables
    long long* path_size  = new long long[Pop_citysize];
    long long** paths      = new long long*[Pop_citysize];
    long long* num_drops   = new long long[Pop_citysize];
    long long*** drops      = new long long**[Pop_citysize];
 
    for (int i = 0; i < Pop_citysize; ++i) {
        const auto& res = results[i];
        path_size[i] = res.path_size;
        paths[i]     = new long long[res.path_size];
        std::copy(res.path,res.path + res.path_size,paths[i]);
        num_drops[i] = res.drops_size;
        drops[i]     = new long long*[res.drops_size];
        for (int j = 0; j < res.drops_size; ++j) {
            drops[i][j] = new long long[3];
            std::copy(res.drops[j],res.drops[j] + 3,drops[i][j]);
        }
    }
    std::ofstream ofs(argv[2]);
    if (!ofs) {
        std::cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }
    for (int i = 0; i < Pop_citysize; ++i) {
        for (int j = 0; j < path_size[i]; ++j) {
            ofs << paths[i][j] << ' ';
        }
        ofs << '\n';
    }
    for (int i = 0; i < Pop_citysize; ++i) {
        for (int j = 0; j < num_drops[i]; ++j) {
            auto triple = drops[i][j];
            ofs << triple[0] << ' '
                << triple[1] << ' '
                << triple[2] << ' ';
        }
        ofs << '\n';
    }
    ofs.close();

    // Free memory ..
    for (int i = 0; i < Pop_citysize; ++i) {
        delete[] paths[i];
        for (int j = 0; j < num_drops[i]; ++j) {
            delete[] drops[i][j];
        }
        delete[] drops[i];
    }
    delete[] drops;
    delete[] num_drops;
    delete[] paths;
    delete[] path_size;
    
    return 0;
}