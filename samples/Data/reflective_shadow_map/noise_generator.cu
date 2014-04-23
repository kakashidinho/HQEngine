#include <cuda.h>
#include <cuda_runtime_api.h>

#include <curand_kernel.h>

#define PI 3.141592654f

surface<void, cudaSurfaceType2D> surfaceWrite;

__global__ void kernel(curandState * randStates) {
	//init rand
	unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1000, id, id, &randStates[id]);

	//get two uniform random floats
	float S1 = curand_uniform(&randStates[id]);
	float S2 = curand_uniform(&randStates[id]);

	//now write to output surface
	ushort4 output;
	output.x = __float2half_rn 	(S1);
	output.y = __float2half_rn(0.5 * sin(2.0 * PI * S2) + 0.5); //scale down to between [0..1]
	output.z = __float2half_rn(0.5 * cos(2.0 * PI * S2) + 0.5); //scale down to between [0..1]
	output.w = __float2half_rn(S1 * S1);

	int x = threadIdx.x;
	int y = blockIdx.x;
	surf2Dwrite(output, surfaceWrite, x * sizeof(ushort4), y);
}


extern "C" void cudaGenerateNoiseMapKernel(cudaArray_t outputArray, unsigned int width, unsigned int height) {
	cudaError_t err;
	curandState * randStates;

	//alloc curand states
	err = cudaMalloc(&randStates, width * height * sizeof(curandState));

	//bind array to global surface
	err = cudaBindSurfaceToArray(surfaceWrite, outputArray);

	//call kernel
	kernel << < width, height >> > (randStates);

	//clean up
	err = cudaFree(randStates);
}