/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#pragma once

#include <vorbis/vorbisfile.h>

namespace HQAudioInternal
{
	void InitVorbisFileCallbacks(ov_callbacks * callbacks);
};
