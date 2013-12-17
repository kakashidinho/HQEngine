/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQAudioPCH.h"
#include "alFunctionPointers.h"

#ifdef WIN32
#	define DLL_HANDLE HMODULE
#	define LoadDll LoadLibraryA
#	define CloseDll FreeLibrary
#	define GetDllFunc GetProcAddress
#endif

#define GET_FUNC_PTR(type, name) name = (type)GetDllFunc(al_lib, #name)

DLL_HANDLE al_lib = NULL;

/*
 * Device Management
 */
LPALCOPENDEVICE fp_alcOpenDevice = NULL;

LPALCCLOSEDEVICE fp_alcCloseDevice = NULL;

/*
 * Context Management
 */
LPALCCREATECONTEXT fp_alcCreateContext = NULL;

LPALCMAKECONTEXTCURRENT fp_alcMakeContextCurrent = NULL;

LPALCDESTROYCONTEXT fp_alcDestroyContext = NULL;

/*
 * Query functions
 */
LPALCGETSTRING fp_alcGetString = NULL;

/* Create Source objects */
LPALGENSOURCES alGenSources = NULL; 

/* Delete Source objects */
LPALDELETESOURCES alDeleteSources = NULL;

/*
 * Error support.
 * Obtain the most recent error generated in the AL state machine.
 */
LPALGETERROR alGetError = NULL;

LPALGETENUMVALUE alGetEnumValue = NULL;

/*
 * Set Listener parameters
 */
LPALLISTENERF alListenerf = NULL;

LPALLISTENER3F alListener3f = NULL;

LPALLISTENERFV alListenerfv = NULL; 

LPALLISTENERI alListeneri = NULL;

LPALLISTENER3I alListener3i = NULL;

LPALLISTENERIV alListeneriv = NULL;

/*
 * Set Source parameters
 */
LPALSOURCEF alSourcef = NULL; 

LPALSOURCE3F alSource3f = NULL;

LPALSOURCEFV alSourcefv = NULL; 

LPALSOURCEI alSourcei = NULL; 

LPALSOURCE3I alSource3i = NULL;

LPALSOURCEIV alSourceiv = NULL;


/*
 * Get Source parameters
 */
LPALGETSOURCEF alGetSourcef = NULL;

LPALGETSOURCE3F alGetSource3f = NULL;

LPALGETSOURCEFV alGetSourcefv = NULL;

LPALGETSOURCEI alGetSourcei = NULL;

LPALGETSOURCE3I alGetSource3i = NULL;

LPALGETSOURCEIV alGetSourceiv = NULL;


/*
 * Source vector based playback calls
 */

/* Play, replay, or resume (if paused) a list of Sources */
LPALSOURCEPLAYV alSourcePlayv = NULL;

/* Stop a list of Sources */
LPALSOURCESTOPV alSourceStopv = NULL;

/* Rewind a list of Sources */
LPALSOURCEREWINDV alSourceRewindv = NULL;

/* Pause a list of Sources */
LPALSOURCEPAUSEV alSourcePausev = NULL;

/*
 * Source based playback calls
 */

/* Play, replay, or resume a Source */
LPALSOURCEPLAY alSourcePlay = NULL;

/* Stop a Source */
LPALSOURCESTOP alSourceStop = NULL;

/* Rewind a Source (set playback postiton to beginning) */
LPALSOURCEREWIND alSourceRewind = NULL;

/* Pause a Source */
LPALSOURCEPAUSE alSourcePause = NULL;

/*
 * Source Queuing 
 */
LPALSOURCEQUEUEBUFFERS alSourceQueueBuffers = NULL;

LPALSOURCEUNQUEUEBUFFERS alSourceUnqueueBuffers = NULL;

/* Create Buffer objects */
LPALGENBUFFERS alGenBuffers = NULL;

/* Delete Buffer objects */
LPALDELETEBUFFERS alDeleteBuffers = NULL;

/* Verify a handle is a valid Buffer */
LPALISBUFFER alIsBuffer = NULL;

/* Specify the data to be copied into a buffer */
LPALBUFFERDATA alBufferData = NULL;
/*
 * Get Buffer parameters
 */
LPALGETBUFFERI alGetBufferi = NULL;

/*-------control speed of sound---------*/
LPALSPEEDOFSOUND alSpeedOfSound = NULL;


bool InitALFunctionPointers()
{
#ifdef WIN32
	al_lib = LoadDll("OpenAL32.dll");
#else
#	error need implement
#endif
	if (al_lib == NULL)
		return false;
	/*
	 * Device Management
	 */
	fp_alcOpenDevice = (LPALCOPENDEVICE )GetDllFunc(al_lib, "alcOpenDevice");

	fp_alcCloseDevice = (LPALCCLOSEDEVICE)GetDllFunc(al_lib, "alcCloseDevice");

	/*
	 * Context Management
	 */
	fp_alcCreateContext = (LPALCCREATECONTEXT)GetDllFunc(al_lib, "alcCreateContext");

	fp_alcMakeContextCurrent = (LPALCMAKECONTEXTCURRENT )GetDllFunc(al_lib, "alcMakeContextCurrent");

	fp_alcDestroyContext = (LPALCDESTROYCONTEXT )GetDllFunc(al_lib, "alcDestroyContext");

	/*
	 * Query functions
	 */
	fp_alcGetString = (LPALCGETSTRING ) GetDllFunc(al_lib, "alcGetString");

	/* Create Source objects */
	GET_FUNC_PTR(LPALGENSOURCES, alGenSources); 

	/* Delete Source objects */
	GET_FUNC_PTR(LPALDELETESOURCES, alDeleteSources);

	/*
	 * Error support.
	 * Obtain the most recent error generated in the AL state machine.
	 */
	GET_FUNC_PTR(LPALGETERROR, alGetError);

	GET_FUNC_PTR(LPALGETENUMVALUE, alGetEnumValue);

	/*
	 * Set Listener parameters
	 */
	GET_FUNC_PTR(LPALLISTENERF, alListenerf);

	GET_FUNC_PTR(LPALLISTENER3F, alListener3f);

	GET_FUNC_PTR(LPALLISTENERFV, alListenerfv); 

	GET_FUNC_PTR(LPALLISTENERI, alListeneri);

	GET_FUNC_PTR(LPALLISTENER3I, alListener3i);

	GET_FUNC_PTR(LPALLISTENERIV, alListeneriv);

	/*
	 * Set Source parameters
	 */
	GET_FUNC_PTR(LPALSOURCEF, alSourcef); 

	GET_FUNC_PTR(LPALSOURCE3F, alSource3f);

	GET_FUNC_PTR(LPALSOURCEFV, alSourcefv); 

	GET_FUNC_PTR(LPALSOURCEI, alSourcei); 

	GET_FUNC_PTR(LPALSOURCE3I, alSource3i);

	GET_FUNC_PTR(LPALSOURCEIV, alSourceiv);


	/*
	 * Get Source parameters
	 */
	GET_FUNC_PTR(LPALGETSOURCEF, alGetSourcef);

	GET_FUNC_PTR(LPALGETSOURCE3F, alGetSource3f);

	GET_FUNC_PTR(LPALGETSOURCEFV, alGetSourcefv);

	GET_FUNC_PTR(LPALGETSOURCEI, alGetSourcei);

	GET_FUNC_PTR(LPALGETSOURCE3I, alGetSource3i);

	GET_FUNC_PTR(LPALGETSOURCEIV, alGetSourceiv);


	/*
	 * Source vector based playback calls
	 */

	/* Play, replay, or resume (if paused) a list of Sources */
	GET_FUNC_PTR(LPALSOURCEPLAYV, alSourcePlayv);

	/* Stop a list of Sources */
	GET_FUNC_PTR(LPALSOURCESTOPV, alSourceStopv);

	/* Rewind a list of Sources */
	GET_FUNC_PTR(LPALSOURCEREWINDV, alSourceRewindv);

	/* Pause a list of Sources */
	GET_FUNC_PTR(LPALSOURCEPAUSEV, alSourcePausev);

	/*
	 * Source based playback calls
	 */

	/* Play, replay, or resume a Source */
	GET_FUNC_PTR(LPALSOURCEPLAY, alSourcePlay);

	/* Stop a Source */
	GET_FUNC_PTR(LPALSOURCESTOP, alSourceStop);

	/* Rewind a Source (set playback postiton to beginning) */
	GET_FUNC_PTR(LPALSOURCEREWIND, alSourceRewind);

	/* Pause a Source */
	GET_FUNC_PTR(LPALSOURCEPAUSE, alSourcePause);

	/*
	 * Source Queuing 
	 */
	GET_FUNC_PTR(LPALSOURCEQUEUEBUFFERS, alSourceQueueBuffers);

	GET_FUNC_PTR(LPALSOURCEUNQUEUEBUFFERS, alSourceUnqueueBuffers);

	/* Create Buffer objects */
	GET_FUNC_PTR(LPALGENBUFFERS, alGenBuffers);

	/* Delete Buffer objects */
	GET_FUNC_PTR(LPALDELETEBUFFERS, alDeleteBuffers);

	/* Verify a handle is a valid Buffer */
	GET_FUNC_PTR(LPALISBUFFER, alIsBuffer);

	/* Specify the data to be copied into a buffer */
	GET_FUNC_PTR(LPALBUFFERDATA, alBufferData);
	
	/*
	 * Get Buffer parameters
	 */
	GET_FUNC_PTR(LPALGETBUFFERI, alGetBufferi);

	/* Speed of Sound */
	GET_FUNC_PTR(LPALSPEEDOFSOUND, alSpeedOfSound);
	return true;
}

void ReleaseALFunctionPointers()
{
	if (al_lib != NULL)
	{
		CloseDll(al_lib);
		al_lib = NULL;
	}
}
