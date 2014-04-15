

#if defined _MSC_VER || defined WIN32 || defined _WIN32
#include <windows.h>	
#include <HQClosedHashTable.h>
#else
#include <pthread.h>
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32

#include "HQCudaBinding.h"



enum GraphicsDeviceType{
	GDT_D3D11,
	GDT_D3D9,
	GDT_GL,
	GDT_UNKNOWN,
};

#if defined _MSC_VER || defined WIN32 || defined _WIN32
class ThreadSpecificDeviceType: private HQClosedHashTable<DWORD, GraphicsDeviceType>
{
public:
	//set graphics device type current for calling thread
	void SetCurrentThreadGraphicsDeviceType(GraphicsDeviceType type) {
		this->Add(ThreadSpecificDeviceType::GetCurrenThreadId(), type);
	}
	
	//get current thread's graphics device's type
	GraphicsDeviceType GetCurrentThreadGraphicsDeviceType() {
		bool found = false;
		GraphicsDeviceType type = this->GetItem(ThreadSpecificDeviceType::GetCurrenThreadId(), found);
		if (!found)
			type = GDT_UNKNOWN;
		return type;
	}
private:
	static DWORD GetCurrenThreadId() { return GetCurrentThreadId();}
};	

#else //pthread version
#pragma message "warning: untested implementation"

static void threadDataDestructor(void *data);

class ThreadSpecificDeviceType
{
public:
	ThreadSpecificDeviceType() {
		pthread_key_create(&threadSpecificKey, threadDataDestructor);
	}
	~ThreadSpecificDeviceType(){
		pthread_key_delete(threadSpecificKey);
	}
	//set graphics device type current for calling thread
	void SetCurrentThreadGraphicsDeviceType(GraphicsDeviceType type) {
		GraphicsDeviceType * threadDevType = (GraphicsDeviceType*)pthread_getspecific(threadSpecificKey);
		if (threadDevType == NULL)
		{
			threadDevType = (GraphicsDeviceType *)malloc(sizeof(GraphicsDeviceType));
			pthread_setspecific(threadSpecificKey, threadDevType);
		}
		
		*threadDevType = type;
	}
	
	//get current thread's graphics device's type
	GraphicsDeviceType GetCurrentThreadGraphicsDeviceType() {
		GraphicsDeviceType type = GDT_UNKNOWN;
		void * threadData = pthread_getspecific(threadSpecificKey);
		
		if (threadData != NULL)
			type = *(GraphicsDeviceType*)threadData;
		
		return type;
	}
	
	static void InvalidateCurrentGraphicsDeviceType()
	{
		pthread_setspecific(threadSpecificKey, NULL);
	}
private:
	pthread_key_t threadSpecificKey;
};

void threadDataDestructor(void *data){
	ThreadSpecificDeviceType::InvalidateCurrentGraphicsDeviceType();
	free(data);
}
#endif//#if defined _MSC_VER || defined WIN32 || defined _WIN32

static ThreadSpecificDeviceType g_threadDeviceType;

namespace HQCudaBinding {
	/*---------implementation---*/
	cudaError_t cudaGraphicsSetDevice(HQRenderDevice* pDevice, int cudaDevice) {
		if (pDevice == NULL)
			return cudaErrorInvalidValue ;
		cudaError_t re;
		const char *deviceDesc = pDevice->GetDeviceDesc();
		if (!strcmp(deviceDesc, "Direct3D11"))
		{
			re = cudaD3D11SetDirect3DDevice(pDevice, cudaDevice);
			if (re == cudaSuccess)
				g_threadDeviceType.SetCurrentThreadGraphicsDeviceType(GDT_D3D11);
		}
		else if (!strcmp(deviceDesc, "Direct3D9"))
		{
			re = cudaD3D9SetDirect3DDevice(pDevice, cudaDevice);
			if (re == cudaSuccess)
				g_threadDeviceType.SetCurrentThreadGraphicsDeviceType(GDT_D3D9);
		}
		else if (!strncmp(deviceDesc, "OpenGL", 6))
		{
			re = cudaGLSetGLDevice(pDevice, cudaDevice);
			if (re == cudaSuccess)
				g_threadDeviceType.SetCurrentThreadGraphicsDeviceType(GDT_GL);
		}
		else
			re = cudaErrorNotYetImplemented;
			
		return re;
	}	
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQTexture* texture, unsigned int flags)
	{
		switch (g_threadDeviceType.GetCurrentThreadGraphicsDeviceType())
		{
		case GDT_D3D11:
			return cudaGraphicsD3D11RegisterResource(cudaResource, texture, flags);
		case GDT_D3D9:
			return cudaGraphicsD3D9RegisterResource(cudaResource, texture, flags);
		case GDT_GL:
			return cudaGraphicsGLRegisterResource(cudaResource, texture, flags);
		default:
			return cudaErrorNotYetImplemented;
		}
	}
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQEngineTextureResource* textureRes, unsigned int flags)
	{
		if (textureRes == NULL)
			return cudaErrorInvalidValue ;
		return cudaGraphicsRegisterResource(cudaResource, textureRes->GetTexture(), flags);
	}
	
	cudaError_t cudaGraphicsRegisterBuffer(struct cudaGraphicsResource **cudaResource, HQGraphicsResourceRawRetrievable* buffer, unsigned int flags)
	{
		switch (g_threadDeviceType.GetCurrentThreadGraphicsDeviceType())
		{
		case GDT_D3D11:
			return cudaGraphicsD3D11RegisterResource(cudaResource, buffer, flags);
		case GDT_D3D9:
			return cudaGraphicsD3D9RegisterResource(cudaResource, buffer, flags);
		case GDT_GL:
			return cudaGraphicsGLRegisterBuffer(cudaResource, buffer, flags);
		default:
			return cudaErrorNotYetImplemented;
		}
	}
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQVertexBuffer* vBuffer, unsigned int flags)
	{
		return cudaGraphicsRegisterBuffer(cudaResource, vBuffer, flags);
	}
	
	cudaError_t cudaGraphicsRegisterResource(struct cudaGraphicsResource **cudaResource, HQIndexBuffer* iBuffer, unsigned int flags)
	{
		return cudaGraphicsRegisterBuffer(cudaResource, iBuffer, flags);
	}
};//HQCudaBinding