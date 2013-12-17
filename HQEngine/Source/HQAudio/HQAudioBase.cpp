/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQAudioPCH.h"
#include "HQAudioBase.h"

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "winstore/HQAudioWinStore.h"

using namespace HQWinStoreFileSystem;
#endif

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#	if defined HQ_STATIC_ENGINE//force link to ogg libs
#		pragma comment(lib, "libvorbis_static.lib")
#		pragma comment(lib, "libvorbisfile.lib")
#		pragma comment(lib, "libogg_static.lib")		
#	endif

#elif defined WIN32
#	if defined _DEBUG || defined DEBUG
#		ifdef _STATIC_CRT
#			pragma comment(lib, "../../VS/Output/Debug static CRT/HQUtilD.lib")
#			pragma comment(lib, "../../VS/Output/StaticDebug static CRT/HQUtilMathD.lib")
#			pragma comment(lib, "../../VS/Libs/Debug static CRT/libvorbis_static.lib")
#			pragma comment(lib, "../../VS/Libs/Debug static CRT/libvorbisfile_static.lib")
#			pragma comment(lib, "../../VS/Libs/Debug static CRT/libogg_static.lib")
#		else
#			pragma comment(lib, "../../VS/Output/Debug/HQUtilD.lib")
#			pragma comment(lib, "../../VS/Output/StaticDebug/HQUtilMathD.lib")
#			pragma comment(lib, "../../VS/Libs/Debug/libvorbis_static.lib")
#			pragma comment(lib, "../../VS/Libs/Debug/libvorbisfile_static.lib")
#			pragma comment(lib, "../../VS/Libs/Debug/libogg_static.lib")
#		endif
#	else
#		ifdef _STATIC_CRT
#			pragma comment(lib, "../../VS/Output/Release static CRT/HQUtil.lib")
#			pragma comment(lib, "../../VS/Output/StaticRelease static CRT/HQUtilMath.lib")
#			pragma comment(lib, "../../VS/Libs/Release static CRT/libvorbis_static.lib")
#			pragma comment(lib, "../../VS/Libs/Release static CRT/libvorbisfile_static.lib")
#			pragma comment(lib, "../../VS/Libs/Release static CRT/libogg_static.lib")
#		else
#			pragma comment(lib, "../../VS/Output/Release/HQUtil.lib")
#			pragma comment(lib, "../../VS/Output/StaticRelease/HQUtilMath.lib")
#			pragma comment(lib, "../../VS/Libs/Release/libvorbis_static.lib")
#			pragma comment(lib, "../../VS/Libs/Release/libvorbisfile_static.lib")
#			pragma comment(lib, "../../VS/Libs/Release/libogg_static.lib")
#		endif
#	endif
#endif

HQBaseAudioDevice * ge_pAudioDevice = NULL;

static ov_callbacks g_vorbis_callback = {
	(size_t (*)(void *, size_t, size_t, void *))  ::fread,
    (int (*)(void *, ogg_int64_t, int))              ::fseek,
    (int (*)(void *))                             ::fclose,
    (long (*)(void *))                            ::ftell
};

/*-----EndiannessChecker-------------*/

struct EndiannessChecker
{
	EndiannessChecker()
	{
		hqubyte bytes[] = {2 , 0 , 0 , 0};
		hquint32 i = *(hquint32*)bytes;
		if (i == 2)
			endianness = 0 ;//little endian
		else
			endianness = 1;//big endian
	}

	int endianness;
};

static const EndiannessChecker endiannessChecker;

/*-------HQRawStreamBufferSupplier-------------*/
class HQRawStreamBufferSupplier: public HQStreamBufferSupplier
{
public:
	HQRawStreamBufferSupplier(HQ_AUDIO_FILE_STREAM *f)
	{
		m_start_position = ftell(f);

		m_file = f;
	}
	virtual ~HQRawStreamBufferSupplier() 
	{
		fclose(m_file);
	}

	virtual size_t GetNextData(void * buffer, size_t bufferSize) 
	{
		return fread(buffer, 1, bufferSize, m_file);
	}

	virtual HQReturnVal Rewind()
	{
		fseek(m_file, m_start_position, SEEK_SET);
		return HQ_OK;
	}

private:
	HQ_AUDIO_FILE_STREAM * m_file;
	size_t m_start_position;
};

/*-------HQVorbisStreamBufferSupplier-------------*/
class HQVorbisStreamBufferSupplier: public HQStreamBufferSupplier
{
public:
	HQVorbisStreamBufferSupplier(OggVorbis_File &f)
	{
		m_file = f;
		
		m_start_position = ov_raw_tell(&m_file);
	}
	virtual ~HQVorbisStreamBufferSupplier() 
	{
		ov_clear(&m_file);
	}

	virtual size_t GetNextData(void * buffer, size_t bufferSize) 
	{
		int bitstream;
		long bytesRead = 0;
		long totalBytes = 0;

		do {
			bytesRead = ov_read(&m_file , (char*)buffer + totalBytes, bufferSize - totalBytes, endiannessChecker.endianness,
					2, 1, &bitstream);

			if (bytesRead > 0)
				totalBytes += bytesRead;
		}while (bytesRead > 0);

		return totalBytes;
	}

	virtual HQReturnVal Rewind()
	{
		//these methods don't work
		//ov_pcm_seek(&m_file, 0);
		//ov_raw_seek(&m_file, m_start_position);
		//ov_time_seek(&m_file, 0.0);

		//have to reopen the file
		HQ_AUDIO_FILE_STREAM * f = (HQ_AUDIO_FILE_STREAM*)m_file.datasource;
		m_file.datasource = NULL;
		rewind(f);
		ov_clear(&m_file);
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		ov_test_callbacks(f, &m_file, NULL, 0, g_vorbis_callback);
#else
		ov_test(f, &m_file, NULL, 0) ;
#endif

		ov_test_open(&m_file);

		return HQ_OK;
	}

private:
	OggVorbis_File m_file;
	size_t m_start_position;
};

/*----------------------*/


/*----------HQBaseAudioDevice--------------------*/
HQBaseAudioDevice::HQBaseAudioDevice(HQLogStream *logStream, const char *logPrefix, bool flushLog)
		: HQLoggableObject(logStream, logPrefix, flushLog)
{
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	HQAudioInternal::InitVorbisFileCallbacks(&g_vorbis_callback);
#endif
}

HQReturnVal HQBaseAudioDevice::Release()
{
	if (ge_pAudioDevice != NULL)
	{
		delete ge_pAudioDevice;
		ge_pAudioDevice = NULL;
		return HQ_OK;
	}

	return HQ_FAILED;
}
bool HQBaseAudioDevice::IsWaveFile(HQ_AUDIO_FILE_STREAM *f)
{
	bool isWave = false;
	hqbyte data[4];
	fread(data, 4 , 1, f);
	if (!strncmp(data, "RIFF" , 4))
	{
		fseek(f, 4 , SEEK_CUR);//chunk size
		fread(data, 4 , 1, f);
		if (!strncmp(data, "WAVE" , 4))
		{
			fread(data, 4 , 1, f);
			if (!strncmp(data, "fmt " , 4))
				isWave = true;
		}
	}
	
	rewind(f);

	return isWave;
}

bool HQBaseAudioDevice::IsVorbisFile(HQ_AUDIO_FILE_STREAM *f, OggVorbis_File &vf)
{
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	if (ov_test_callbacks(f, &vf, NULL, 0, g_vorbis_callback) == 0 )
#else
	if (ov_test(f, &vf, NULL, 0) == 0 )
#endif
		return true;
	ov_clear(&vf);
	return false;
}

void HQBaseAudioDevice::ReleaseAudioDataInfo(AudioDataInfo &info)
{
	if (info.data != NULL)
	{
		free(info.data);
		info.data = NULL;
	}
}

bool HQBaseAudioDevice::IsAudioFileLoaded(const char *fileName, hquint32 *existID)
{
	if (fileName == NULL)
		return false;
	/*-------check if this file already loaded---------*/
	HQItemManager<HQBaseAudioBuffer>::Iterator ite;
	m_bufferManager.GetIterator(ite);
	while(!ite.IsAtEnd())
	{
		if (ite->m_audioFileName != NULL && strcmp(ite->m_audioFileName, fileName) == 0)
		{
			if (existID != NULL)
				*existID = ite.GetID();
			return true;
		}
		++ite;
	}

	return false;
}

HQReturnVal HQBaseAudioDevice::GetWaveInfo(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioInfo &info)
{
	hqint32 chunkSize, subChunkSize[2];
	hqint32 data32;//32 bit data
	hqshort16 data16;//16 bit data
	hqshort16 blockAlign;
	hqbyte data8[4];

	fseek(f, 4, SEEK_SET);//to chunk size
	fread(&chunkSize, 4, 1, f);//read chunk size
	fseek(f, 16, SEEK_SET);//to sub chunk 1's size
	fread(&subChunkSize[0], 4, 1, f);//read chunk size
	fread(&data16, 2 , 1 , f);
	if (data16 != 1)//must be 1 (PCM)
	{
		Log("Error : File \"%s\" is not supported!", fileName);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	fread(&data16, 2, 1, f);//read num channels
	info.channels = data16;
	if (!this->IsMultiChannelsSupported(info.channels))//this multi channels type not supported
	{
		Log("Error : File \"%s\" is not supported because it contains %d channels audio!", fileName, info.channels);
		return  HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	fread(&data32, 4, 1, f);//read sample rate
	info.sampleRate = data32;
	fseek(f, 4, SEEK_CUR);//byte rate ignored
	fread(&blockAlign, 2, 1, f);//block align

	fread(&data16, 2, 1, f);//read bits per channel
	info.bits = data16;
	
	fread(data8, 4, 1, f);//read subchunk 2 id, must be 'data'
	
	if (strncmp(data8, "data", 4) != 0 && strncmp(data8, "atad", 4) != 0)
	{
		//try to get next 2 bytes
		fseek(f, - 2, SEEK_CUR);
		fread(data8, 4, 1, f);//read subchunk 2 id, must be 'data'
		if (strncmp(data8, "data", 4) != 0 && strncmp(data8, "atad", 4) != 0)
		{
			Log("Error : File \"%s\" is not supported!", fileName);
			return HQ_FAILED_FORMAT_NOT_SUPPORT;
		}
	}
	
	fread(&subChunkSize[1], 4, 1, f);//data size
	info.size = subChunkSize[1];
	info.samples = info.size / blockAlign;//number of samples

	return HQ_OK;
}

HQReturnVal HQBaseAudioDevice::GetWaveData(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioDataInfo &info)
{
	info.data = NULL;
	
	HQReturnVal re = GetWaveInfo(fileName, f, info);
	if (HQFailed(re))
		return re;

	info.data = (hqbyte*)malloc(info.size);
	if (info.data == NULL)
		return HQ_FAILED_MEM_ALLOC;
	fread(info.data, info.size, 1, f);//read data

	return HQ_OK;
}

HQReturnVal HQBaseAudioDevice::CreateWaveStreamBufferSupplier(const char *fileName, HQ_AUDIO_FILE_STREAM *f, AudioInfo &info, HQStreamBufferSupplier*& supplierOut)
{
	HQReturnVal re = GetWaveInfo(fileName, f, info);
	if (HQFailed(re))
		return re;

	supplierOut = HQ_NEW HQRawStreamBufferSupplier(f);

	return HQ_OK;
}


HQReturnVal HQBaseAudioDevice::GetVorbisInfo(const char *fileName, OggVorbis_File &vf, AudioInfo &info)
{
	info.size = 0;
	info.bits = 16;

	if (ov_test_open(&vf) != 0)
	{
		Log("Error : File \"%s\" is not supported!", fileName);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	
	vorbis_info *vinfo = ov_info(&vf , -1);
	info.channels = vinfo->channels;
	info.sampleRate = vinfo->rate;//sampling rate
	

	HQReturnVal re = HQ_OK;

	if (this->IsMultiChannelsSupported(info.channels))
	{
		//get decoded size
		info.samples = ov_pcm_total(&vf, -1);//total samples
		info.size = (size_t)(info.channels * info.bits / 8 * info.samples);
	}
	else//this multi channels type not supported
	{
		Log("Error : File \"%s\" is not supported because it contains %d channels audio!", fileName, info.channels);
		re =  HQ_FAILED_FORMAT_NOT_SUPPORT;
	}

	return re;

}

HQReturnVal HQBaseAudioDevice::DecodeVorbis(const char *fileName, OggVorbis_File &vf, AudioDataInfo &info)
{
	info.data = NULL;	

	HQReturnVal re = GetVorbisInfo(fileName, vf, info);
	if (!HQFailed(re))
	{
		info.data = (hqbyte*)malloc(info.size);
		if (info.data == NULL)
		{
			re = HQ_FAILED_MEM_ALLOC;
		}
		else
		{
			//decode data into buffer
			long bytes;
			int bitstream;
			int bytesDone = 0;
			do
			{
				bytes = ov_read(&vf , info.data + bytesDone, info.size - bytesDone, endiannessChecker.endianness,
					2, 1, &bitstream);
				bytesDone += bytes;
			}while(bytes > 0);
#if defined DEBUG || defined _DEBUG
			int a = 0;//debug break
#endif
		}//successfully allocate memory
	}//if (!HQFailed(re))

	ov_clear(&vf);

	return re;
}

HQReturnVal HQBaseAudioDevice::CreateVorbisStreamBufferSupplier(const char *fileName, OggVorbis_File &vf, AudioInfo &info, HQStreamBufferSupplier*& supplierOut)
{
	HQReturnVal re = GetVorbisInfo(fileName, vf, info);

	if (re == HQ_OK)
		supplierOut = HQ_NEW HQVorbisStreamBufferSupplier(vf);

	return re;
}


#ifdef RAW_VORBIS
///Get Compressed vorbis file data without decode it. {fileName} is used for logging
HQReturnVal HQBaseAudioDevice::GetVorbisData(const char *fileName, OggVorbis_File &vf, AudioDataInfo &info)
{
	info.data = NULL;
	info.size = 0;
	info.bits = 16;

	if (ov_test_open(&vf) != 0)
	{
		Log("Error : File \"%s\" is not supported!", fileName);
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}
	
	info.samples = ov_pcm_total(&vf, -1);//total samples

	vorbis_info *vinfo = ov_info(&vf , -1);
	info.channels = vinfo->channels;
	info.sampleRate = vinfo->rate;//sampling rate
	
	FILE *f = (FILE*)vf.datasource;
	
	ov_clear(&vf);

	rewind(f);
	
	if (!this->IsMultiChannelsSupported(info.channels))//this multi channels type not supported
	{
		Log("Error : File \"%s\" is not supported because it contains %d channels audio!", fileName, info.channels);
		return  HQ_FAILED_FORMAT_NOT_SUPPORT;
	}


	//get raw data
	fseek(f, 0, SEEK_END);
	info.size = ftell(f);
	rewind(f);
	
	info.data = (hqbyte*)malloc(info.size);
	if (info.data == NULL)
	{
		return HQ_FAILED_MEM_ALLOC;
	}
	
	fread(info.data, info.size, 1, f);//read file data
	rewind(f);
	
	return HQ_OK;
}
#endif

#ifdef WIN32
extern HQBaseAudioDevice * HQCreateXAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate);
#endif
extern HQBaseAudioDevice * HQCreateAudioDeviceAL(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate);


HQAudioDevice * HQCreateAudioDevice(hqfloat32 speedOfSound, HQLogStream *logStream, bool flushLog, bool leftHandedCoordinate)
{
	if (ge_pAudioDevice == NULL)
	{
#if !defined HQ_NO_XAUDIO && (defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		//use xaudio2 first
		ge_pAudioDevice = HQCreateXAudioDevice(speedOfSound, logStream, flushLog, leftHandedCoordinate);
		if (ge_pAudioDevice == NULL)//failed , now try openAL
#endif

#if !(defined HQ_NO_OPEN_AL || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)//windows store app does not support openAL
			ge_pAudioDevice = HQCreateAudioDeviceAL(speedOfSound, logStream, flushLog, leftHandedCoordinate);
#else
			return NULL;
#endif
	}

	return ge_pAudioDevice;
}
