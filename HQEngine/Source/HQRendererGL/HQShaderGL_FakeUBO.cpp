/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "../HQLinkedList.h"
#include "../HQClosedStringHashTable.h"
#include "glHeaders.h"
#include "HQShaderGL_FakeUBO.h"

#define MAX_UNIFORM_BUFFER_SLOTS 36

#ifndef min
#define min (a, b) (a < b ? a : b)
#endif



/*---------HQFakeUniformBufferGL--------------*/
HQFakeUniformBufferGL::HQFakeUniformBufferGL(hq_uint32 size, bool isDynamic)
: boundSlots(HQ_NEW HQPoolMemoryManager(sizeof(BufferSlotList::LinkedListNodeType), MAX_UNIFORM_BUFFER_SLOTS))
{
	//must allocate a buffer with size is multiple of 16 byte
	size_t realSize = size;
	size_t remain = size % 16;
	if (remain > 0)
		realSize += (16 - remain);
	this->pRawBuffer = HQ_NEW hqubyte8[realSize]; 

	this->size = size;
	this->isDynamic = isDynamic;
}
HQFakeUniformBufferGL::~HQFakeUniformBufferGL()
{
	delete[] (hqubyte8*)pRawBuffer;
}

/*----------------------HQFakeUniformBlkElem----------------------*/
//element of fake uniform buffer
struct HQFakeUniformBlkElem {
	HQFakeUniformBlkElem(GLuint program, const HQUniformBlkElemInfoGL& parsed_info);
	~HQFakeUniformBlkElem();

	size_t ConsumeData(void *data, size_t dataSize);//return number of bytes consumed
private:
	GLint *index;//index of each element in array
	GLuint arraySize; 
	size_t arrayElemSize;//size of data that one array element will consume, not actual size of array element in shader
	size_t totalSize;//total size of this element

	typedef void (HQFakeUniformBlkElem::*ConsumeFuncPtrType) (const void *, GLuint);
	ConsumeFuncPtrType ComsumeFunc;

	void DontConsume(const void * data, GLuint numElems) {}

	void ConsumeInt(const void * data, GLuint numElems);
	void ConsumeInt2(const void * data, GLuint numElems);
	void ConsumeInt3(const void * data, GLuint numElems);
	void ConsumeInt4(const void * data, GLuint numElems);

	void ConsumeFloat(const void * data, GLuint numElems);
	void ConsumeFloat2(const void * data, GLuint numElems);
	void ConsumeFloat3(const void * data, GLuint numElems);
	void ConsumeFloat4(const void * data, GLuint numElems);

	void ConsumeMatrix2(const void * data, GLuint numElems);
	void ConsumeMatrix3(const void * data, GLuint numElems);
	void ConsumeMatrix4(const void * data, GLuint numElems);
	void ConsumeMatrix2x3(const void * data, GLuint numElems);
	void ConsumeMatrix2x4(const void * data, GLuint numElems);
	void ConsumeMatrix3x2(const void * data, GLuint numElems);
	void ConsumeMatrix3x4(const void * data, GLuint numElems);
	void ConsumeMatrix4x2(const void * data, GLuint numElems);
	void ConsumeMatrix4x3(const void * data, GLuint numElems);
};


/*-----------------HQShaderProgramFakeUBO-------------------*/
struct HQShaderProgramFakeUBO : public HQBaseShaderProgramGL
{
	struct BufferSlotInfo{
		hquint32 slot;
		HQLinkedList<HQSharedPtr<HQFakeUniformBlkElem> > constants;//list of uniforms that consume datat from this buffer slot
	};

	typedef HQLinkedList<BufferSlotInfo > BufferSlotList;
	BufferSlotList uBufferSlotList;//list of uniform buffer slots associated with this shader program
};

/*---------------HQBaseShaderManagerGL_FakeUBO::BufferSlotInfo------------*/
struct HQBaseShaderManagerGL_FakeUBO::BufferSlotInfo {
	hquint32 dirtyFlags;
	HQSharedPtr<HQFakeUniformBufferGL> buffer;
	HQFakeUniformBufferGL::BufferSlotList::LinkedListNodeType* bufferLink;//this link can be used for fast removing the slot from buffer's bound slots
};


/*--------------------------HQFakeUniformBlkElem implementation---------------------------------*/
HQFakeUniformBlkElem::HQFakeUniformBlkElem(GLuint program, const HQUniformBlkElemInfoGL& parsed_info)
: index(NULL)
{
	//this is size of data that one array element will consume, not actual size of array element in shader
	this->arrayElemSize = parsed_info.row * 4 * sizeof(hqfloat32);
	//this is number of elements in array
	this->arraySize = parsed_info.arraySize;

	this->totalSize = this->arrayElemSize * this->arraySize;

	if (parsed_info.col == 4)
	{
		//only first element is enough
		this->index = HQ_NEW GLint[1];
		this->index[0] = glGetUniformLocation(program, parsed_info.name.c_str());
	}
	else
	{
		//get every element
		this->index = HQ_NEW GLint[this->arraySize];
		char *name = HQ_NEW char[parsed_info.name.size() + 11];
		for (GLuint i = 0; i < this->arraySize; ++i)
		{
			sprintf(name, "%s[%d]", parsed_info.name.c_str(), i);
			this->index[i] = glGetUniformLocation(program, name);
		}

		delete[] name;
	}

	this->ComsumeFunc = &HQFakeUniformBlkElem::DontConsume;

	if (parsed_info.row == 1)
	{
		switch (parsed_info.col)
		{
		case 1:
			this->ComsumeFunc = parsed_info.integer ? &HQFakeUniformBlkElem::ConsumeInt : &HQFakeUniformBlkElem::ConsumeFloat;
			break;
		case 2:
			this->ComsumeFunc = parsed_info.integer ? &HQFakeUniformBlkElem::ConsumeInt2 : &HQFakeUniformBlkElem::ConsumeFloat2;
			break;
		case 3:
			this->ComsumeFunc = parsed_info.integer ? &HQFakeUniformBlkElem::ConsumeInt3 : &HQFakeUniformBlkElem::ConsumeFloat3;
			break;
		case 4:
			this->ComsumeFunc = parsed_info.integer ? &HQFakeUniformBlkElem::ConsumeInt4 : &HQFakeUniformBlkElem::ConsumeFloat4;
			break;
		}//switch (parsed_info.col)
	}//if (parsed_info.row == 1)
	else
	{
		switch (parsed_info.row)
		{
		case 2:
			switch (parsed_info.col)
			{
			case 2:
				this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix2;
				break;
			case 3:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix2x3;
				break;
			case 4:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix2x4;
				break;
			}
			break;
		case 3:
			switch (parsed_info.col)
			{
			case 2:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix3x2;
				break;
			case 3:
				this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix3;
				break;
			case 4:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix3x4;
				break;
			}
			break;
		case 4:
			switch (parsed_info.col)
			{
			case 2:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix4x2;
				break;
			case 3:
				if (GLEW_VERSION_2_1)
					this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix4x3;
				break;
			case 4:
				this->ComsumeFunc = &HQFakeUniformBlkElem::ConsumeMatrix4;
				break;
			}
			break;
		}//switch (parsed_info.row)
	}//else of if (parsed_info.row == 1)
}

HQFakeUniformBlkElem::~HQFakeUniformBlkElem()
{
	SafeDeleteArray(index);
}

inline size_t HQFakeUniformBlkElem::ConsumeData(void *data, size_t dataSize)
{

	GLuint numElems;

	size_t consumeSize ;

	if (this->totalSize > dataSize)
	{
		numElems = (GLuint)(dataSize / this->arrayElemSize);
		consumeSize = numElems *  this->arrayElemSize;
	}
	else
	{
		numElems = this->arraySize;
		consumeSize = this->totalSize;
	}

	(this->*ComsumeFunc) (data, numElems);

	return consumeSize;
}

void HQFakeUniformBlkElem::ConsumeInt(const void * data, GLuint numElems)
{
	const int vectorElem = 1;
	const int vectorSize = vectorElem * sizeof(hqint32);
	hqint32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform1iv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeInt2(const void * data, GLuint numElems)
{
	const int vectorElem = 2;
	const int vectorSize = vectorElem * sizeof(hqint32);
	hqint32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform2iv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeInt3(const void * data, GLuint numElems)
{
	const int vectorElem = 3;
	const int vectorSize = vectorElem * sizeof(hqint32);
	hqint32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform3iv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeInt4(const void * data, GLuint numElems)
{
	glUniform4iv(this->index[0], numElems, (const GLint*)data);
}

void HQFakeUniformBlkElem::ConsumeFloat(const void * data, GLuint numElems)
{
	const int vectorElem = 1;
	const int vectorSize = vectorElem * sizeof(hqfloat32);
	hqfloat32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform1fv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeFloat2(const void * data, GLuint numElems)
{
	const int vectorElem = 2;
	const int vectorSize = vectorElem * sizeof(hqfloat32);
	hqfloat32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform2fv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeFloat3(const void * data, GLuint numElems)
{
	const int vectorElem = 3;
	const int vectorSize = vectorElem * sizeof(hqfloat32);
	hqfloat32 vector[vectorElem];
	for (GLuint i = 0; i < numElems; ++i)
	{
		memcpy(vector, (hqubyte8*)data + this->arrayElemSize * i, vectorSize);
		glUniform3fv(this->index[i], 1, vector);
	}
}
void HQFakeUniformBlkElem::ConsumeFloat4(const void * data, GLuint numElems)
{
	glUniform4fv(this->index[0], numElems, (const GLfloat*)data);
}

void HQFakeUniformBlkElem::ConsumeMatrix2(const void * data, GLuint numElems)
{
	const int matrixRow = 2;
	const int matrixCol = 2;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix2fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix3(const void * data, GLuint numElems)
{
	const int matrixRow = 3;
	const int matrixCol = 3;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix3fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix4(const void * data, GLuint numElems)
{
	glUniformMatrix4fv(this->index[0], numElems, GL_FALSE, (const GLfloat*)data);
}
void HQFakeUniformBlkElem::ConsumeMatrix2x3(const void * data, GLuint numElems)
{
	const int matrixRow = 2;
	const int matrixCol = 3;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix2x3fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix2x4(const void * data, GLuint numElems)
{
	const int matrixRow = 2;
	const int matrixCol = 4;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix2x4fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix3x2(const void * data, GLuint numElems)
{
	const int matrixRow = 3;
	const int matrixCol = 2;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix3x2fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix3x4(const void * data, GLuint numElems)
{
	glUniformMatrix3x4fv(this->index[0], numElems, GL_FALSE, (const GLfloat*)data);
}
void HQFakeUniformBlkElem::ConsumeMatrix4x2(const void * data, GLuint numElems)
{
	const int matrixRow = 4;
	const int matrixCol = 2;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix4x2fv(this->index[i], 1, GL_FALSE, matrix);
	}
}
void HQFakeUniformBlkElem::ConsumeMatrix4x3(const void * data, GLuint numElems)
{
	const int matrixRow = 4;
	const int matrixCol = 3;
	const int matrixRowSize = matrixCol * sizeof(hqfloat32);
	const size_t fullVectorSize = 4 * sizeof(hqfloat32);
	hqfloat32 matrix[matrixRow * matrixCol];


	int dataRow = 0;
	GLuint i = 0;
	for (; i < numElems; ++i)
	{
		for (int r = 0; r < matrixRow; ++r, ++dataRow)
		{
			//since each row will consume one vector
			memcpy((hqubyte8*)matrix + r * matrixRowSize, (hqubyte8*)data + fullVectorSize * dataRow, matrixRowSize);
		}
		glUniformMatrix4x3fv(this->index[i], 1, GL_FALSE, matrix);
	}
}

/*--------------HQBaseShaderManagerGL_FakeUBO-----------*/
HQBaseShaderManagerGL_FakeUBO::HQBaseShaderManagerGL_FakeUBO(HQLogStream* logFileStream, const char * logPrefix, bool flushLog)
:HQBaseCommonShaderManagerGL(logFileStream, logPrefix, flushLog)
{
	this->uBufferSlots = HQ_NEW BufferSlotInfo[MAX_UNIFORM_BUFFER_SLOTS];
}

HQBaseShaderManagerGL_FakeUBO::~HQBaseShaderManagerGL_FakeUBO()
{
	delete[] this->uBufferSlots;
}

HQBaseShaderProgramGL* HQBaseShaderManagerGL_FakeUBO::CreateNewProgramObject()
{
	return HQ_NEW HQShaderProgramFakeUBO();
}

void HQBaseShaderManagerGL_FakeUBO::OnProgramCreated(HQBaseShaderProgramGL *program)
{
	HQShaderProgramFakeUBO* programFakeUBO = static_cast<HQShaderProgramFakeUBO*> (program);
	//create uniform block elements

	HQClosedStringHashTable<bool> uniformTable;//this table ensures no duplicated uniform variables

	const int numShaders = 3;
	HQSharedPtr<HQShaderObjectGL> shaderObs[numShaders];

	shaderObs[0] = this->shaderObjects.GetItemPointer(program->vertexShaderID);
	shaderObs[1] = this->shaderObjects.GetItemPointer(program->geometryShaderID);
	shaderObs[2] = this->shaderObjects.GetItemPointer(program->pixelShaderID);

	for (int i = 0; i < numShaders; ++i)
	{
		if (shaderObs[i] != NULL && shaderObs[i]->pUniformBlocks != NULL)
		{
			//inspect through every uniform blocks of every attached shader objects
			HQLinkedList<HQUniformBlockInfoGL>::Iterator block_ite;
			
			for (shaderObs[i]->pUniformBlocks->GetIterator(block_ite); !block_ite.IsAtEnd(); ++block_ite)
			{
				if (block_ite->index >= MAX_UNIFORM_BUFFER_SLOTS)
					continue;
				HQLinkedList<HQUniformBlkElemInfoGL>::Iterator uniform_ite;
				HQShaderProgramFakeUBO::BufferSlotInfo newBufferSlotInfo;
				newBufferSlotInfo.slot = block_ite->index;
				//for each element
				for (block_ite->blockElems.GetIterator(uniform_ite); !uniform_ite.IsAtEnd(); ++uniform_ite)
				{
					bool found = false;
					uniformTable.GetItem(uniform_ite->name, found);
					if (!found && glGetUniformLocation(program->programGLHandle, uniform_ite->name.c_str()) != -1)
					{
						uniformTable.Add(uniform_ite->name, true);
						HQFakeUniformBlkElem *newElem = HQ_NEW HQFakeUniformBlkElem(program->programGLHandle, *uniform_ite);

						//add uniform to buffer slot dependent list
						newBufferSlotInfo.constants.PushBack(newElem);
					}

				}//for (block_ite->blockElems.GetIterator(uniform_ite); !uniform_ite.IsAtEnd(); ++uniform_ite)

				//add associated buffer slot to shader program
				if (newBufferSlotInfo.constants.GetSize() > 0)
					programFakeUBO->uBufferSlotList.PushBack(newBufferSlotInfo);

			}//for (shaderObs[i]->pUniformBlocks->GetIterator(block_ite); !block_ite.IsAtEnd(); ++block_ite)

			SafeDelete(shaderObs[i]->pUniformBlocks);//no more need for this list
		}//if (shaderObs[i] != NULL && shaderObs[i]->pUniformBlocks != NULL)

	}//for (int i = 0; i < numShaders; ++i)
}


void HQBaseShaderManagerGL_FakeUBO::OnProgramActivated(HQBaseShaderProgramGL* program)
{
	HQShaderProgramFakeUBO* programFakeUBO = static_cast<HQShaderProgramFakeUBO*> (program);
	//mark every associated buffer slots as dirty for this shader program
	HQShaderProgramFakeUBO::BufferSlotList::Iterator slot_ite;
	for (programFakeUBO->uBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	{
		this->uBufferSlots[slot_ite->slot].dirtyFlags = 1;
	}
}

void HQBaseShaderManagerGL_FakeUBO::Commit()
{
	HQShaderProgramFakeUBO * program = static_cast<HQShaderProgramFakeUBO*> (this->GetItemRawPointer(this->activeProgram));
	if (program != NULL)
	{
		//check for dirty constant buffer slot
		HQShaderProgramFakeUBO::BufferSlotList::Iterator slot_ite;
		//for each constant buffer
		for (program->uBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
		{
			BufferSlotInfo &bufferSlot = this->uBufferSlots[slot_ite->slot];
			HQFakeUniformBufferGL *constBuffer = bufferSlot.buffer.GetRawPointer();

			if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
			{
				//this slot is dirty. need to update constant data
				hqubyte8 * pData = (hqubyte8*)constBuffer->pRawBuffer;
				hquint32 offset = 0;
				//for each constant
				HQLinkedList<HQSharedPtr<HQFakeUniformBlkElem> >::Iterator const_ite;
				for (slot_ite->constants.GetIterator(const_ite);
					!const_ite.IsAtEnd() && offset < constBuffer->size;
					++const_ite)
				{
					offset += (*const_ite)->ConsumeData(pData + offset, constBuffer->size - offset);
				}

				bufferSlot.dirtyFlags = 0;//mark as not dirty

			}//if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
		}//for (program->uBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	}//if (program != NULL)
}

HQReturnVal HQBaseShaderManagerGL_FakeUBO::CreateUniformBuffer(hq_uint32 size, void *initData, bool isDynamic, hq_uint32 *pBufferIDOut)
{
	HQFakeUniformBufferGL* pNewBuffer = HQ_NEW HQFakeUniformBufferGL(size, isDynamic);
	if (initData != NULL)
		memcpy(pNewBuffer->pRawBuffer, initData, size);
	if (!this->uniformBuffers.AddItem(pNewBuffer, pBufferIDOut))
	{
		HQ_DELETE(pNewBuffer);
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}
HQReturnVal HQBaseShaderManagerGL_FakeUBO::DestroyUniformBuffer(hq_uint32 bufferID)
{
	return (HQReturnVal)this->uniformBuffers.Remove(bufferID);
}
void HQBaseShaderManagerGL_FakeUBO::DestroyAllUniformBuffers()
{
	this->uniformBuffers.RemoveAll();
}
HQReturnVal HQBaseShaderManagerGL_FakeUBO::SetUniformBuffer(hq_uint32 slot, hq_uint32 bufferID)
{
	if (slot >= MAX_UNIFORM_BUFFER_SLOTS)
		return HQ_FAILED;
	HQSharedPtr<HQFakeUniformBufferGL> buffer = this->uniformBuffers.GetItemPointer(bufferID);
	BufferSlotInfo *pBufferSlot = this->uBufferSlots + slot;
	if (pBufferSlot->buffer != buffer)
	{
		if (pBufferSlot->buffer != NULL)
		{
			pBufferSlot->buffer->boundSlots.RemoveAt(pBufferSlot->bufferLink);//remove the link with the old buffer
		}
		pBufferSlot->buffer = buffer;
		pBufferSlot->bufferLink = buffer->boundSlots.PushBack(slot);

		pBufferSlot->dirtyFlags = 1;//notify all dependent shaders
	}

	return HQ_FAILED;
}
HQReturnVal HQBaseShaderManagerGL_FakeUBO::MapUniformBuffer(hq_uint32 bufferID, void **ppData)
{
	HQFakeUniformBufferGL* pBuffer = uniformBuffers.GetItemRawPointer(bufferID);

#if defined _DEBUG || defined DEBUG

	if (pBuffer == NULL)
		return HQ_FAILED;

	if (!ppData)
		return HQ_FAILED;
#endif

	*ppData = pBuffer->pRawBuffer;

	return HQ_OK;
}
HQReturnVal HQBaseShaderManagerGL_FakeUBO::UnmapUniformBuffer(hq_uint32 bufferID)
{
	HQFakeUniformBufferGL* pBuffer = uniformBuffers.GetItemRawPointer(bufferID);
#if defined _DEBUG || defined DEBUG	

	if (pBuffer == NULL)
		return HQ_FAILED;
#endif

	HQFakeUniformBufferGL::BufferSlotList::Iterator ite;
	for (pBuffer->boundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		BufferSlotInfo *pBufferSlot = this->uBufferSlots + (*ite);
		pBufferSlot->dirtyFlags = 1;//mark buffer slot as dirty
	}

	return HQ_OK;
}

HQReturnVal HQBaseShaderManagerGL_FakeUBO::UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)
{
	HQFakeUniformBufferGL* pBuffer = this->uniformBuffers.GetItemRawPointer(bufferID);
#if defined _DEBUG || defined DEBUG	
	if (pBuffer == NULL)
		return HQ_FAILED;
	if (pBuffer->isDynamic == true)
	{
		this->Log("Error : dynamic buffer can't be updated using UpdateUniformBuffer method!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

	memcpy(pBuffer->pRawBuffer, pData, pBuffer->size);

	HQFakeUniformBufferGL::BufferSlotList::Iterator ite;
	for (pBuffer->boundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		BufferSlotInfo *pBufferSlot = this->uBufferSlots + (*ite);
		pBufferSlot->dirtyFlags = 1;//mark buffer slot as dirty
	}

	return HQ_OK;
}
