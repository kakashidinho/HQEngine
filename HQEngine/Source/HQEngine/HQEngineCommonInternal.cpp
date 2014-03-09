/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "HQEngineCommonInternal.h"
#include "../HQDataStream.h"

namespace HQEngineHelper
{

void seek_datastream (void* fileHandle, long offset, int origin)
{
	HQDataReaderStream* stream = (HQDataReaderStream*) fileHandle;
	stream->Seek(offset, (HQDataReaderStream::StreamSeekOrigin)origin);
}

size_t tell_datastream (void* fileHandle)
{
	HQDataReaderStream* stream = (HQDataReaderStream*) fileHandle;
	return (size_t)stream->Tell();
}

size_t read_datastream ( void * ptr, size_t size, size_t count, void * fileHandle )
{
	HQDataReaderStream* stream = (HQDataReaderStream*) fileHandle;
	return stream->ReadBytes(ptr, size, count);
}

/*-------------for easy porting from standard C io---------------*/
int fgetc(HQDataReaderStream *stream)
{
	return stream->GetByte();
}

size_t fread ( void * ptr, size_t size, size_t count, HQDataReaderStream * stream )
{
	return stream->ReadBytes(ptr, size, count);
}
int fclose ( HQDataReaderStream * stream )
{
	if (stream == NULL)
		return 0;
	stream->Release();
	return 0;
}

long ftell ( HQDataReaderStream * stream )
{
	return stream->Tell();
}

int fseek ( HQDataReaderStream * stream, long int offset, int origin )
{
	return stream->Seek(offset, (HQDataReaderStream::StreamSeekOrigin)origin);
}

void rewind( HQDataReaderStream *stream)
{
	stream->Rewind();
}


};//namespace HQEngineHelper