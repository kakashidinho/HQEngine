/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#pragma once

#include "../../HQEngineCommon.h"
#include "../../HQSharedPointer.h"
#include "../../HQDataStream.h"

#define USE_WIN32_HANDLE 1

namespace HQWinStoreFileSystem
{
	class ReadDataResult
	{
	public:
		ReadDataResult(unsigned char*pData, unsigned int dataSize);
		~ReadDataResult();

		const unsigned char* getData() const {return pData;}
		unsigned int getDataSize() const {return dataSize;}
	private:
		unsigned char* pData;
		unsigned int dataSize;
	};

#if USE_WIN32_HANDLE
	class HQENGINE_API BufferedDataReader: public HQDataReaderStream
	{
	public:
		BufferedDataReader(const char *fileName, bool useExactFileName = false);
		~BufferedDataReader();

		void Release() {delete this;}

		int GetByte();
		size_t ReadBytes(void* dataOut, size_t elemSize, size_t elemCount);
		int Close();
		int Seek(long long offset, StreamSeekOrigin origin);
		void Rewind();
		long long Tell() const;//ftell implemetation

		size_t TotalSize() const {return totalSize;}
		bool Good() const {return Tell() < TotalSize();}

		bool GetLine(char * buffer, size_t maxBufferSize);//maxBufferSize includes terminating character

		const char *GetName() const {return name;}
	private:
		bool ReadBytes(unsigned char *ptr, size_t bytes);

		HANDLE fileHandle;

		LARGE_INTEGER currentPointer;
		size_t totalSize;//total size of stream from the start to the end

		char * name;//file name
	};
#else
	//async version
	class HQENGINE_API BufferedDataReader: public HQDataReaderStream
	{
	public:
		BufferedDataReader(Windows::Storage::Streams::IRandomAccessStream ^stream, const char *_name);
		~BufferedDataReader();

		void Release() {delete this;}

		int GetByte();
		size_t ReadBytes(void* dataOut, size_t elemSize, size_t elemCount);
		int Close();
		int Seek(long long offset, StreamSeekOrigin origin);
		void Rewind();
		long long Tell() const;//ftell implemetation

		size_t TotalSize() const {return totalSize;}
		bool Good() const {return Tell() < TotalSize();}

		bool GetLine(char * buffer, size_t maxBufferSize);//maxBufferSize includes terminating character
		
		const char *GetName() const {return name;}
	private:
		void ResetStream(size_t position);
		void ReadBytes(unsigned char *ptr, size_t bytes);
		bool FillBuffer();

		Windows::Storage::Streams::IRandomAccessStream ^stream; 
		Windows::Storage::Streams::DataReader ^reader;

		Platform::Array<unsigned char>^ buffer;
		size_t bufferSize;
		size_t unconsumedBufferLength;
		size_t bufferOffset;
		size_t streamOffset;
		size_t totalSize;//total size of stream from the start to the end

		char * name;//file name
	};
#endif

	//this will search file in installed folder, local folder, roaming folder, temp folder of the app
	HQENGINE_API Windows::Storage::StorageFile ^ OpenFile(const char *file);
	//this will open file using exact name specified by <file>. No concatenate  with current directory
	HQENGINE_API Windows::Storage::StorageFile ^ OpenExactFile(const char *fileName);
	//open or create file in local folder
	HQENGINE_API Windows::Storage::StorageFile ^ OpenOrCreateFile(const char *file);
	//open or create file in local folder
	HQENGINE_API Windows::Storage::Streams::DataWriter ^ OpenOrCreateFileForWrite(const char *file);

	//this will search file in installed folder, local folder, roaming folder, temp folder of the app
	HQENGINE_API BufferedDataReader * OpenFileForRead(const char *file);

	//this will open a file using the exact name specified by <file>. No concatenate  with current directory
	HQENGINE_API BufferedDataReader * OpenExactFileForRead(const char *file);

	//*ppDataOut must be release by calling delete[]
	HQENGINE_API bool ReadData(const char *fileName, unsigned char *&pDataOut, unsigned int &pDataSizeOut);

	HQENGINE_API HQSharedPtr<ReadDataResult> ReadData(const char *fileName);

	//emulating current directory
	const wchar_t * GetCurrentDirConst();
	wchar_t * GetCurrentDir();//returned pointer should be deleted
	void SetCurrentDir(const wchar_t *dir);
	void SetCurrentDir(const char *dir);
};
