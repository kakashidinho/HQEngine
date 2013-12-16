#pragma once

#include <vorbis/vorbisfile.h>

namespace HQAudioInternal
{
	void InitVorbisFileCallbacks(ov_callbacks * callbacks);
};