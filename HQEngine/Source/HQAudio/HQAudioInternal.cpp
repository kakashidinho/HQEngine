/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQAudioPCH.h"
#include "HQAudioInternal.h"

#include "../HQDataStream.h"

namespace HQAudioInternal
{
	//vorbis file callbacks
	static size_t read_func  (void *ptr, size_t size, size_t nmemb, void *datasource)
	{
		if (datasource == NULL)
			return 0;

		return ((HQDataReaderStream*)datasource)->ReadBytes(ptr, size, nmemb);
	}

	static int    seek_func (void *datasource, long long offset, int whence)
	{
		if (datasource == NULL)
			return EOF;
		return ((HQDataReaderStream*)datasource)->Seek(offset, (HQDataReaderStream::StreamSeekOrigin) whence);
	}

	static int    close_func (void *datasource)
	{
		if (datasource == NULL)
			return EOF;

		((HQDataReaderStream*)datasource)->Release();

		return 0;
	}

	static long   tell_func  (void *datasource)
	{
		if (datasource == NULL)
			return EOF;

		return ((HQDataReaderStream*)datasource)->Tell();
	}

	void InitVorbisFileCallbacks(ov_callbacks * callbacks)
	{
		callbacks->read_func = read_func;
		callbacks->seek_func = seek_func;
		callbacks->close_func = close_func;
		callbacks->tell_func = tell_func;
	}
};
