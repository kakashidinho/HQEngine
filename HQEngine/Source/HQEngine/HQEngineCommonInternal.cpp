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


char * GetContainingDir(const char *file)//returned pointer should to be deleted
{
	const char *fileNameStart = strrchr(file, '/');
#ifdef WIN32
	if (fileNameStart == NULL)
		fileNameStart = strrchr(file, '\\');
#endif

	if (fileNameStart == NULL)
		return NULL;

	long dirlen = fileNameStart - file;
	char *dir = HQ_NEW char[dirlen + 1];
	strncpy(dir, file, dirlen);
	dir[dirlen] = '\0';

	return dir;
}

void SetCurrentDir(const char *dir)
{
#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
	HQWinStoreFileSystem::SetCurrentDir(dir);
#elif defined WIN32
	SetCurrentDirectoryA(dir);
#else
	chdir(dir);
#endif
}

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
wchar_t * GetCurrentDir()//returned pointer should be deleted
{
	return HQWinStoreFileSystem::GetCurrentDir();
}
void SetCurrentDir(const wchar_t *dir)
{
	HQWinStoreFileSystem::SetCurrentDir(dir);
}

#elif defined WIN32
wchar_t * GetCurrentDir()//returned pointer should be deleted
{
	long cwdSize;
	wchar_t *cwd;
	cwdSize = GetCurrentDirectory(0, NULL) + 1;
	cwd = HQ_NEW wchar_t[cwdSize];

	GetCurrentDirectory(cwdSize, cwd);

	return cwd;
}
void SetCurrentDir(const wchar_t *dir)
{
	SetCurrentDirectory(dir);
}
#else
char * GetCurrentDir()//returned pointer should to be deleted
{
	long cwdSize = pathconf(".", _PC_PATH_MAX);
	char *cwd = HQ_NEW char[cwdSize];
	
	getcwd(cwd, cwdSize);

	return cwd;
}
#endif

};//namespace HQEngineHelper