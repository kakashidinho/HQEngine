/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "ImgByteStream.h"

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

using namespace HQWinStoreFileSystem;

#endif

#include <string.h>

ImgByteStream::ImgByteStream()
: isMemoryMode(false),
streamSize(0),
memory(NULL),
iterator(0)
{
}
ImgByteStream::~ImgByteStream()
{
	Clear();
}

void ImgByteStream::CreateByteStreamFromMemory(const hqubyte8 * memory, size_t size)
{
	Clear();
	this->memory = memory;
	this->streamSize = size;
	this->isMemoryMode = true;
}
bool ImgByteStream::CreateByteStreamFromFile(const char *fileName)
{
	Clear();

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
	file = HQWinStoreFileSystem::OpenFileForRead(fileName);
#else
	file = fopen(fileName, "rb");
#endif
	if (file == NULL)
		return false;

	fseek(file , 0, SEEK_END);
	this->streamSize = ftell(file);
	rewind(file);

	this->isMemoryMode = false;
	return true;
}

void ImgByteStream::Rewind()
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
			rewind(file);
	}
	else if (memory != NULL)
		iterator = 0;
}

void ImgByteStream::Seek(size_t pos)
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
			fseek(file, pos, SEEK_SET);
	}
	else if (memory != NULL)
		iterator = pos;
}

void ImgByteStream::Advance(size_t offset)
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
			fseek(file, offset, SEEK_CUR);
	}
	else if (memory != NULL)
	{
		iterator += offset;
		if (iterator >= streamSize)
			iterator = streamSize - 1;
	}
}

hqint32 ImgByteStream::GetByte()
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
			return fgetc(file);
	}
	else if (memory != NULL)
	{
		if (iterator >= streamSize)
			return EOF;
		return memory[iterator ++];
	}
	return EOF;
}

bool ImgByteStream::GetBytes(void *bytes, size_t size)
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
		{
			if (fread(bytes, size, 1, file)!= 1)
				return false;
		}
		else return false;
	}
	else if (memory != NULL)
	{
		if (streamSize - iterator < size)
			return false;
		memcpy(bytes, memory + iterator, size);
		iterator += size;
	}
	else return false;

	return true;
}

size_t ImgByteStream::TryGetBytes(void *buffer, size_t elemSize, size_t numElems)
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
		{
			return fread(buffer, elemSize, numElems, file);
		}
		else return 0;
	}
	else if (memory != NULL)
	{
		size_t readElems = 0;

		for (size_t i = 0; i < numElems; ++i)
		{
			if (streamSize - iterator < elemSize)
				break;
			memcpy((hqbyte8*)buffer + i * elemSize, memory + iterator, elemSize);
			iterator += elemSize;

			readElems++;
		}

		return readElems;
	}

	return 0;
}

void ImgByteStream::Clear()
{
	if (!this->isMemoryMode)
	{
		if (file != NULL)
		{
			fclose(file);
			file = NULL;
		}
	}
	else
	{
		this->memory = NULL;
		this->iterator = 0;
	}
	
	this->isMemoryMode = false;
	this->streamSize = 0;

}
