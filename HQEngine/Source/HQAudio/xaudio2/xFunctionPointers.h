/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#pragma once

#include <XAudio2.h>
#include <X3DAudio.h>

typedef void (*fpX3DAudioInitialize_t) (UINT32 SpeakerChannelMask, FLOAT32 SpeedOfSound, __out X3DAUDIO_HANDLE Instance);
typedef void (*fpX3DAudioCalculate_t) (__in const X3DAUDIO_HANDLE Instance, __in const X3DAUDIO_LISTENER* pListener, __in const X3DAUDIO_EMITTER* pEmitter, UINT32 Flags, __inout X3DAUDIO_DSP_SETTINGS* pDSPSettings);

extern fpX3DAudioInitialize_t fpX3DAudioInitialize;
extern fpX3DAudioCalculate_t fpX3DAudioCalculate;

#define X3DAudioInitialize fpX3DAudioInitialize
#define X3DAudioCalculate fpX3DAudioCalculate

bool InitXAudioFunctions();
void ReleaseXAudioFunctions();
