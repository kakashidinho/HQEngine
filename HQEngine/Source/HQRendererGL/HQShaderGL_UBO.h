#ifndef HQ_SHADER_GL_UBO_H
#define HQ_SHADER_GL_UBO_H

#include "HQShaderGL_Common.h"

#ifndef GLES

#define MAX_UNIFORM_BUFFER_SLOTS 36

class HQBaseShaderManagerGL_UBO;

struct HQUniformBufferGL
{
	HQUniformBufferGL(HQBaseShaderManagerGL_UBO *manager , hq_uint32 size ,GLenum usage) ;
	~HQUniformBufferGL();
	
	GLuint buffer;
	HQBaseShaderManagerGL_UBO *manager;
	hq_uint32 size;
	GLenum usage;
};

//for openGL 3.1 and later or ARB_uniform_buffer_object extension
class HQBaseShaderManagerGL_UBO : public HQBaseCommonShaderManagerGL
{
protected:
	HQItemManager<HQUniformBufferGL> uniformBuffers;

	GLuint currentBoundUBuffer;
	HQSharedPtr<HQUniformBufferGL> uBufferSlots[MAX_UNIFORM_BUFFER_SLOTS];

	inline void BindUniformBuffer(GLuint buffer)
	{
		if (currentBoundUBuffer != buffer)
		{
			glBindBuffer(GL_UNIFORM_BUFFER , buffer);
			currentBoundUBuffer = buffer;
		}
	}

public:
	HQBaseShaderManagerGL_UBO(HQLogStream* logFileStream , const char * logPrefix , bool flushLog);
	
	GLuint GetCurrentBoundUBuffer() {return this->currentBoundUBuffer;}
	void InvalidateCurrentBoundUBuffer() {this->currentBoundUBuffer = 0;}

	HQReturnVal CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut);
	HQReturnVal DestroyUniformBuffer(hq_uint32 bufferID);
	void DestroyAllUniformBuffers();
	HQReturnVal SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID );
	HQReturnVal MapUniformBuffer(hq_uint32 bufferID , void **ppData);
	HQReturnVal UnmapUniformBuffer(hq_uint32 bufferID);
	HQReturnVal UpdateUniformBuffer(hq_uint32 bufferID, const void * pData);
};


#endif

#endif