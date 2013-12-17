/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "HQAudioWinStore.h"

#include "../../HQEngine/winstore/HQWinStoreFileSystem.h"

namespace HQAudioInternal
{
	//vorbis file callbacks
	static size_t read_func  (void *ptr, size_t size, size_t nmemb, void *datasource)
	{
		if (datasource == NULL)
			return 0;

		return ((HQWinStoreFileSystem::BufferedDataReader*)datasource)->ReadBytes(ptr, size, nmemb);
	}

	static int    seek_func (void *datasource, long long offset, int whence)
	{
		if (datasource == NULL)
			return EOF;
		return ((HQWinStoreFileSystem::BufferedDataReader*)datasource)->Seek(offset, (HQDataReaderStream::StreamSeekOrigin) whence);
	}

	static int    close_func (void *datasource)
	{
		if (datasource == NULL)
			return EOF;

		int re = ((HQWinStoreFileSystem::BufferedDataReader*)datasource)->Close();

		delete ((HQWinStoreFileSystem::BufferedDataReader*)datasource);

		return re;
	}

	static long   tell_func  (void *datasource)
	{
		if (datasource == NULL)
			return EOF;

		return ((HQWinStoreFileSystem::BufferedDataReader*)datasource)->Tell();
	}

	void InitVorbisFileCallbacks(ov_callbacks * callbacks)
	{
		callbacks->read_func = read_func;
		callbacks->seek_func = seek_func;
		callbacks->close_func = close_func;
		callbacks->tell_func = tell_func;
	}
};
