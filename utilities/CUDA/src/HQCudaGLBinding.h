#ifndef HQ_CUDA_GL_BINDING_H
#define HQ_CUDA_GL_BINDING_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <HQRenderDevice.h>
#include <HQEngineResManager.h>

#ifndef GL_TEXTURE_CUBE_MAP
#	define GL_TEXTURE_CUBE_MAP 0x8513
#endif

namespace HQCudaBinding {
	cudaError_t cudaGLSetGLDevice(HQRenderDevice* pDevice, int cudaDevice = -1);
	
	cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags);
	
	cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQTexture* texture, unsigned int flags);
	
	cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQVertexBuffer* vBuffer, unsigned int flags);
	
	cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQIndexBuffer* iBuffer, unsigned int flags);
	
	cudaError_t cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* buffer, unsigned int flags);

	/*---------implementation---*/
	inline cudaError_t cudaGLSetGLDevice(HQRenderDevice* pDevice, int cudaDevice) {
		return cudaSuccess;
	}	
	
	inline cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQTexture* texture, unsigned int flags)
	{
		if (texture == NULL)
			return cudaErrorInvalidValue ;
#if defined (__LP64__) || defined(_M_X64) || defined(__amd64__)
		GLuint glHandle = (GLuint)((hquint64)texture->GetRawHandle() & 0xffffffff);
#else
		GLuint glHandle = (hquint32)texture->GetRawHandle();
#endif
		GLenum textureTarget;
		switch (texture->GetType()){
		case HQ_TEXTURE_2D:
			textureTarget = GL_TEXTURE_2D;
		break;
		case HQ_TEXTURE_CUBE:
			textureTarget = GL_TEXTURE_CUBE_MAP;
		break;
		default:
			return cudaErrorNotYetImplemented ;
		}
		
		return cudaGraphicsGLRegisterImage(cudaResource, glHandle, textureTarget, flags);
	}
	
	inline cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags)
	{
		if (textureRes == NULL)
			return cudaErrorInvalidValue ;
		return cudaGraphicsGLRegisterResource(cudaResource, textureRes->GetTexture(), flags);
	}
	
	inline cudaError_t cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* buffer, unsigned int flags)
	{
		if (buffer == NULL)
			return cudaErrorInvalidValue ;
#if defined (__LP64__) || defined(_M_X64) || defined(__amd64__)
		GLuint glHandle = (GLuint)((hquint64)buffer->GetRawHandle() & 0xffffffff);
#else
		GLuint glHandle = (hquint32)buffer->GetRawHandle();
#endif			
		return cudaGraphicsGLRegisterBuffer(cudaResource, glHandle, flags);
	}
	
	inline cudaError_t cudaGraphicsGLRegisterResource(struct cudaGraphicsResource **cudaResource, HQGraphicsBufferRawRetrievable* buffer, unsigned int flags)
	{
		return cudaGraphicsGLRegisterBuffer(cudaResource, buffer, flags);
	}
	
};//namespace HQCudaBinding


#endif