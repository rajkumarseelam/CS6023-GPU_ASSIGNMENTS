# GPU Programming Assignments (CS6023)

This repository contains CUDA implementations for four GPU programming assignments focusing on parallel computing concepts and applications.

## Assignment 1: Parallel Image Processing

### Overview
Implementation of two image transformation filters using CUDA parallel processing:

1. **Inverted Gray Scale Transformation**
   - Converts RGB images to grayscale using averaging
   - Inverts the resulting grayscale image
   - Formula: `gray(i,j) = ⌊(R + G + B) / 3⌋`
   - Final matrix: `result(i,j) = gray(rows-i, j)`

2. **Thomas Transformation**
   - Custom mathematical transformation on RGB channels
   - Formula: `result(i,j) = ⌊0.5 × red(i,j)⌋ + ⌊√green(i,j)⌋ + blue(i,j)`
   - Maintains data type consistency with floor operations

### Technical Implementation
- Parallel processing of image matrices
- Efficient memory transfers between host and device
- Kernel optimization for matrix operations
- Grid and block dimension calculations for various image sizes

## Assignment 2: Parallel 2D Convolution

### Overview
GPU-accelerated 2D convolution operation for image processing with custom filter implementations.

### Key Features
- Handles multi-channel images (H × W × C)
- Processes multiple filters simultaneously (K filters)
- Implements zero-padding for boundary conditions
- Stride length: 1

### Technical Implementation
- **Memory Optimization**: Utilizes shared memory for improved performance
- **Coalesced Memory Access**: Optimized memory access patterns
- **Dynamic Shared Memory**: Adaptable to different filter sizes
- **Parallel Filter Application**: Simultaneous processing of multiple filters
- **Channel-wise Convolution**: Separate convolution per channel with summation

### Algorithm Details
- Input transformation: H × W × C → (H × C) × W
- Filter transformation: R × S × C → (R × C) × S
- Output dimensions: H × W per filter

## Assignment 3: Parallel Graph Algorithms - Route Optimization

### Overview
Implementation of parallel graph algorithms to find the most cost-effective network of pathways with terrain-based cost modifiers.

### Problem Specifications
- Weighted graph with cost modifiers:
  - Green zones: Base cost × 2
  - Traffic areas: Base cost × 5  
  - Department zones: Base cost × 3
- Objective: Minimize total construction cost

### Implemented Algorithms
- **Parallel MST Algorithms**:
  - Borůvka's Algorithm (primary implementation)
  - Support for Kruskal's and Prim's variants
- **Union-Find Data Structure**:
  - Path compression optimization
  - Parallel union operations with atomic CAS
- **Edge Weight Processing**: Parallel weight computation with terrain modifiers

### Technical Features
- Lock-free parallel union operations
- Atomic operations for thread safety
- Efficient component merging
- Shared memory utilization for weight comparisons

## Assignment 4: Large-Scale Evacuation Simulation

### Overview
Complex evacuation planning system using parallel algorithms to optimize civilian movement from populated cities to shelters under various constraints.

### Problem Complexity
- Graph scale: Up to 100,000 cities and 1,000,000 roads
- Population types: Prime-age and elderly (with distance constraints)
- Constraints: Road capacities, shelter capacities, elderly travel limits

### Implementation Strategy

#### Small Graphs (≤ 1000 nodes)
- **Parallel Dijkstra's Algorithm**: For optimal path finding
- **Deterministic approach**: Comprehensive shortest path calculations
- **Memory allocation**: Per-thread distance and predecessor arrays

#### Large Graphs (> 1000 nodes)
- **Randomized Path Finding**: Monte Carlo approach for scalability
- **CUDA Random Number Generation**: Using cuRAND for path selection
- **Heuristic-based decisions**: Balancing computation time and solution quality

### Key Features
1. **Dynamic Path Planning**: Adaptive routing based on current conditions
2. **Capacity Management**: Tracking and updating shelter/road usage
3. **Priority Systems**: Population-based and city-ID based priorities
4. **Penalty Calculations**: Overcrowding management at shelters
5. **Elderly Constraints**: Special handling for distance-limited populations

### Technical Innovations
- Atomic operations for concurrent shelter capacity updates
- Shared memory for efficient path reconstruction
- Coalesced memory access patterns
- Stream-based parallel execution for different population groups

## Common Technical Elements

### CUDA Optimization Techniques
- Grid and block dimension optimization
- Memory coalescing strategies
- Shared memory utilization
- Atomic operations for thread-safe updates
- Warp-level primitives where applicable

### Memory Management
- Efficient host-device memory transfers
- Minimized global memory access
- Strategic use of constant memory (Assignment 2)
- Dynamic memory allocation for variable-sized problems

### Performance Considerations
- Block size optimization (typically 256-1024 threads)
- Occupancy maximization
- Memory bandwidth optimization
- Reduction operations for aggregation

## Build and Execution

### Compilation
```bash
# Basic compilation
nvcc -O3 assignment.cu -o assignment

# With debugging symbols
nvcc -g -G -O3 assignment.cu -o assignment

# Specific GPU architecture (e.g., for Volta)
nvcc -arch=sm_70 -O3 assignment.cu -o assignment

# With all warnings enabled
nvcc -Werror all-warnings -O3 assignment.cu -o assignment

# For profiling with nvprof/nsight
nvcc -lineinfo -O3 assignment.cu -o assignment

# Assignment-specific examples:
# Assignment 1 (Image Processing)
nvcc -O3 CS24M042.cu -o image_transform

# Assignment 2 (Convolution - uses shared memory)
nvcc -O3 -maxrregcount=32 CS24M042.cu -o convolution

# Assignment 3 (Graph algorithms)
nvcc -O3 -arch=sm_50 CS24M042.cu -o route_optimization

# Assignment 4 (Large scale - uses cuRAND)
nvcc -O3 -lcurand CS24M042.cu -o evacuation

```
### Execution

```bash

# Standard execution with input/output redirection
./assignment < input.txt > output.txt

# With error checking
./assignment < input.txt > output.txt 2> error.log

# Performance profiling
nvprof ./assignment < input.txt > output.txt

# Detailed profiling with metrics
nvprof --metrics all ./assignment < input.txt > output.txt

# Memory checking
cuda-memcheck ./assignment < input.txt > output.txt

# Execution on specific GPU (multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 ./assignment < input.txt > output.txt

```

