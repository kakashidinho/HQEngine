#ifndef HQ_CUDA_D3D9_BINDING_H
#define HQ_CUDA_D3D9_BINDING_H
#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined _MSC_VER || defined WIN32 || defined _WIN32
#include <cuda_d3d9_interop.h>
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32

#include <HQRenderDevice.h>
#include <HQEngineResManager.h>

namespace HQCudaBinding {
	cudaError_t cudaD3D9SetDirect3DDevice(HQRenderDevice* pDevice, int cudaDevice = -1);
	
	cudaError_t cudaGraphicsD3D9RegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags);
	
	cudaError_t cudaGraphicsD3D9RegisterResource(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* hqResource, unsigned int flags);
	
	/*---------implementation---*/
	inline cudaError_t cudaD3D9SetDirect3DDevice(HQRenderDevice* pDevice, int cudaDevice) {
		if (pDevice == NULL)
			return cudaErrorInvalidValue;
#if defined _MSC_VER || defined WIN32 || defined _WIN32
		return cudaD3D9SetDirect3DDevice((IDirect3DDevice9*)pDevice->GetRawHandle(), cudaDevice);
#else
		return cudaErrorNotYetImplemented;
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32
	}	
	
	inline cudaError_t cudaGraphicsD3D9RegisterResource(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* hqResource, unsigned int flags)
	{
		if (hqResource == NULL)
			return cudaErrorInvalidValue ;
#if defined _MSC_VER || defined WIN32 || defined _WIN32
		return cudaGraphicsD3D9RegisterResource(cudaResource, (IDirect3DResource9*)hqResource->GetRawHandle(), flags);
#else
		return cudaErrorNotYetImplemented;
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32
	}
	
	inline cudaError_t cudaGraphicsD3D9RegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags)
	{
		if (textureRes == NULL)
			return cudaErrorInvalidValue ;
		return cudaGraphicsD3D9RegisterResource(cudaResource, textureRes->GetTexture(), flags);
	}
	
};//namespace HQCudaBinding


#endif