#ifndef HQ_AL_FUNC_PTR_H
#define HQ_AL_FUNC_PTR_H

#define AL_NO_PROTOTYPES
#include <al.h>
#include <alc.h>


/*
 * Device Management
 */
extern LPALCOPENDEVICE fp_alcOpenDevice;
#define alcOpenDevice fp_alcOpenDevice

extern LPALCCLOSEDEVICE fp_alcCloseDevice;
#define alcCloseDevice fp_alcCloseDevice

/*
 * Context Management
 */
extern LPALCCREATECONTEXT fp_alcCreateContext;
#define alcCreateContext fp_alcCreateContext

extern LPALCMAKECONTEXTCURRENT fp_alcMakeContextCurrent;
#define alcMakeContextCurrent fp_alcMakeContextCurrent

extern LPALCDESTROYCONTEXT fp_alcDestroyContext;
#define alcDestroyContext fp_alcDestroyContext

/*
 * Query functions
 */
extern LPALCGETSTRING fp_alcGetString;
#define alcGetString fp_alcGetString

/* Create Source objects */
extern LPALGENSOURCES alGenSources; 

/* Delete Source objects */
extern LPALDELETESOURCES alDeleteSources;

/*
 * Error support.
 * Obtain the most recent error generated in the AL state machine.
 */
extern LPALGETERROR alGetError;

extern LPALGETENUMVALUE alGetEnumValue;

/*
 * Set Listener parameters
 */
extern LPALLISTENERF alListenerf;

extern LPALLISTENER3F alListener3f;

extern LPALLISTENERFV alListenerfv; 

extern LPALLISTENERI alListeneri;

extern LPALLISTENER3I alListener3i;

extern LPALLISTENERIV alListeneriv;

/*
 * Set Source parameters
 */
extern LPALSOURCEF alSourcef; 

extern LPALSOURCE3F alSource3f;

extern LPALSOURCEFV alSourcefv; 

extern LPALSOURCEI alSourcei; 

extern LPALSOURCE3I alSource3i;

extern LPALSOURCEIV alSourceiv;


/*
 * Get Source parameters
 */
extern LPALGETSOURCEF alGetSourcef;

extern LPALGETSOURCE3F alGetSource3f;

extern LPALGETSOURCEFV alGetSourcefv;

extern LPALGETSOURCEI alGetSourcei;

extern LPALGETSOURCE3I alGetSource3i;

extern LPALGETSOURCEIV alGetSourceiv;


/*
 * Source vector based playback calls
 */

/* Play, replay, or resume (if paused) a list of Sources */
extern LPALSOURCEPLAYV alSourcePlayv;

/* Stop a list of Sources */
extern LPALSOURCESTOPV alSourceStopv;

/* Rewind a list of Sources */
extern LPALSOURCEREWINDV alSourceRewindv;

/* Pause a list of Sources */
extern LPALSOURCEPAUSEV alSourcePausev;

/*
 * Source based playback calls
 */

/* Play, replay, or resume a Source */
extern LPALSOURCEPLAY alSourcePlay;

/* Stop a Source */
extern LPALSOURCESTOP alSourceStop;

/* Rewind a Source (set playback postiton to beginning) */
extern LPALSOURCEREWIND alSourceRewind;

/* Pause a Source */
extern LPALSOURCEPAUSE alSourcePause;

/*
 * Source Queuing 
 */
extern LPALSOURCEQUEUEBUFFERS alSourceQueueBuffers;

extern LPALSOURCEUNQUEUEBUFFERS alSourceUnqueueBuffers;

/* Create Buffer objects */
extern LPALGENBUFFERS alGenBuffers;

/* Delete Buffer objects */
extern LPALDELETEBUFFERS alDeleteBuffers;

/* Verify a handle is a valid Buffer */
extern LPALISBUFFER alIsBuffer;

/* Specify the data to be copied into a buffer */
extern LPALBUFFERDATA alBufferData;

/*
 * Get Buffer parameters
 */
extern LPALGETBUFFERI alGetBufferi;

/*-------control speed of sound---------*/
extern LPALSPEEDOFSOUND alSpeedOfSound;

bool InitALFunctionPointers();
void ReleaseALFunctionPointers();


#endif