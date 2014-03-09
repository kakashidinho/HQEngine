/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_AUDIO_INTERNAL_H
#define HQ_AUDIO_INTERNAL_H

#include <vorbis/vorbisfile.h>

namespace HQAudioInternal
{
	void InitVorbisFileCallbacks(ov_callbacks * callbacks);
};

#endif
