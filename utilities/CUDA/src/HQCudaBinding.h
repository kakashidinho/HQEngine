#ifndef HQ_CUDA_BINDING_H
#define HQ_CUDA_BINDING_H

#include "HQCudaGLBinding.h"
#include "HQCudaD3D9Binding.h"
#include "HQCudaD3D11Binding.h"

#include <string.h>

namespace HQCudaBinding {
	//set graphics device current for calling thread
	cudaError_t cudaGraphicsSetDevice(HQRenderDevice* pDevice, int cudaDevice = -1);
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags);
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQTexture* texture, unsigned int flags);
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQVertexBuffer* vBuffer, unsigned int flags);
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQIndexBuffer* iBuffer, unsigned int flags);
	
	cudaError_t cudaGraphicsRegisterBuffer(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* buffer, unsigned int flags);
	
};

#endif