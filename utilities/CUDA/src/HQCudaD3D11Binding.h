#ifndef HQ_CUDA_D3D11_BINDING_H
#define HQ_CUDA_D3D11_BINDING_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined _MSC_VER || defined WIN32 || defined _WIN32
#include <cuda_d3d11_interop.h>
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32

#include <HQRenderDevice.h>
#include <HQEngineResManager.h>

namespace HQCudaBinding {
	cudaError_t cudaD3D11SetDirect3DDevice(HQRenderDevice* pDevice, int cudaDevice = -1);
	cudaError_t cudaGraphicsD3D11RegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags);
	cudaError_t cudaGraphicsD3D11RegisterResource(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* hqResource, unsigned int flags);
	
	/*---------implementation---*/
	inline cudaError_t cudaD3D11SetDirect3DDevice(HQRenderDevice* pDevice, int cudaDevice) {
		return cudaSuccess;
	}	
	
	inline cudaError_t cudaGraphicsD3D11RegisterResource(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* hqResource, unsigned int flags)
	{
		if (hqResource == NULL)
			return cudaErrorInvalidValue ;
#if defined _MSC_VER || defined WIN32 || defined _WIN32		
		return cudaGraphicsD3D11RegisterResource(cudaResource, (ID3D11Resource*)hqResource->GetRawHandle(), flags);
#else
		return cudaErrorNotYetImplemented;
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32
	}
	
	inline cudaError_t cudaGraphicsD3D11RegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags)
	{
		if (textureRes == NULL)
			return cudaErrorInvalidValue ;
		return cudaGraphicsD3D11RegisterResource(cudaResource, textureRes->GetTexture(), flags);
	}
};//namespace HQCudaBinding

#endif