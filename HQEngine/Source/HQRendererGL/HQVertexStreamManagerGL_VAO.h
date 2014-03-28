/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _VERTEX_STREAM_MAN_GL_VAO_
#define _VERTEX_STREAM_MAN_GL_VAO_

#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQVertexStreamManagerGL.h"

struct HQVertexStreamGL_VAO {
	HQVertexStreamGL_VAO();

	hquint32 HashCode() const;
	bool operator != (const HQVertexStreamGL_VAO &rhs) const;
	GLuint fromWhichBO;//from which VBO this stream fetch data
	hquint32 stride;
};

struct HQVertexArrayObjGL {
	struct Params{
		Params(hquint32 maxNumAttribs, HQVertexInputLayoutGL *vInputLayout);
		Params(const Params& src);
		~Params();

		hquint32 CalculateHashCode();
		hquint32 HashCode() const { return hashCode; }
		bool Equal(const Params* params2) const;

		HQVertexInputLayoutGL *vInputLayout;
		HQVertexStreamGL_VAO *streamSources;
		GLuint indexBufferGL;
		hquint32 maxNumAttribs;
		hquint32 hashCode;
	};

	HQVertexArrayObjGL(const Params &params);
	~HQVertexArrayObjGL();

	bool IsBufferUsed(const HQSharedPtr<HQBufferGL> &vBuffer);

	GLuint nameGL;
	hquint32 dirtyFlags;
	Params params;
	HQClosedPrimeHashTable<GLuint, bool> usedBuffers;//list of VBO used by this VAO
};

class HQVertexStreamManagerGL_VAO : public HQVertexStreamManagerGL{
public:
	HQVertexStreamManagerGL_VAO(hq_uint32 maxVertexAttribs, HQLogStream *logFileStream, bool flushLog);
	~HQVertexStreamManagerGL_VAO();

	void Commit();

	HQReturnVal MapIndexBuffer(hq_uint32 indexBufferID, HQMapType mapType, void **ppData);
	HQReturnVal UnmapIndexBuffer(hq_uint32 indexBufferID);

	HQReturnVal UpdateIndexBuffer(hq_uint32 indexBufferID, hq_uint32 offset, hq_uint32 size, const void * pData);

	HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID, hq_uint32 streamIndex, hq_uint32 stride);
	HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID);
	HQReturnVal SetIndexBuffer(hq_uint32 indexBufferID );

	HQReturnVal RemoveVertexBuffer(hq_uint32 vertexBufferID);
	HQReturnVal RemoveVertexInputLayout(hq_uint32 inputLayoutID);
	HQReturnVal RemoveIndexBuffer(hq_uint32 indexBufferID) ;
	void RemoveAllVertexBuffer();
	void RemoveAllVertexInputLayout();
	void RemoveAllIndexBuffer() ;
private:
	void BindIndexBuffer(GLuint ibo);

	HQSharedPtr<HQVertexArrayObjGL> GetOrCreateNewVAO();
	void ActiveVAO(const HQSharedPtr<HQVertexArrayObjGL> &vao);
	void RemoveDependentVAOs(const HQSharedPtr<HQBufferGL> &vBuffer);//remove all VAOs that use this VBO
	void RemoveDependentVAOs(const HQSharedPtr<HQVertexInputLayoutGL> &vLayout);//remove all VAOs that use this input layout
	void RemoveDependentVAOs(const HQSharedPtr<HQIndexBufferGL> &vBuffer);//remove all VAOs that use this IBO

	GLuint m_defaultVAO_GL;
	HQSharedPtr<HQVertexArrayObjGL> m_currentVAO;
	HQVertexArrayObjGL::Params m_currentVAOParams;
	bool m_currentVAOParamsChanged;
	typedef HQClosedPtrKeyHashTable<HQVertexArrayObjGL::Params*, HQSharedPtr<HQVertexArrayObjGL> > VAOTableType;
	VAOTableType m_vertexArrayObjects;
};

#endif