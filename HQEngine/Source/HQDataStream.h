/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DATA_STREAM_H
#define HQ_DATA_STREAM_H

#include "HQPrimitiveDataType.h"

class HQDataReaderStream
{
public:
	enum StreamSeekOrigin
	{
		BEGIN = 0,
		CURRENT = 1,
		END = 2
	};

	virtual void Release() = 0;//close the stream and release the object

	virtual hqint32 GetByte()= 0 ;///get next byte, return -1 if error
	virtual size_t ReadBytes(void* dataOut, size_t elemSize, size_t elemCount)= 0 ;///returns number of successfully read elements
	virtual hqint32 Seek(hqint64 offset, StreamSeekOrigin origin)= 0 ;///returns non-zero on error
	virtual void Rewind()= 0 ;
	virtual hqint64 Tell() const= 0 ;///same as ftell. return number of bytes between the current stream position and the beginning

	virtual size_t TotalSize() const = 0 ;
	virtual bool Good() const = 0 ;///stil not reach end of stream

protected:
	virtual ~HQDataReaderStream() {}
};


#endif
