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

	HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID, hq_uint32 streamIndex, hq_uint32 stride);
	HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID);
	HQReturnVal SetIndexBuffer(HQIndexBuffer* indexBufferID );

	HQReturnVal RemoveVertexBuffer(HQVertexBuffer* vertexBufferID);
	HQReturnVal RemoveVertexInputLayout(HQVertexLayout* inputLayoutID);
	HQReturnVal RemoveIndexBuffer(HQIndexBuffer* indexBufferID);
	void RemoveAllVertexBuffer();
	void RemoveAllVertexInputLayout();
	void RemoveAllIndexBuffer();

	void BindIndexBuffer(GLuint ibo);//internal use only
private:
	virtual HQVertexBufferGL * CreateNewVertexBufferObj(hq_uint32 size, GLenum usage);
	virtual HQIndexBufferGL * CreateNewIndexBufferObj(hq_uint32 size, GLenum usage, HQIndexDataType dataType);

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