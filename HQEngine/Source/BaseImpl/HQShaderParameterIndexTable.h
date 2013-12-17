/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_PARAM_INDEX_TABLE_H
#define HQ_SHADER_PARAM_INDEX_TABLE_H

#include "../HQClosedHashTable.h"

class HQShaderParamIndexTable : public HQClosedStringHashTable<hq_uint32>
{
protected:
	hq_uint32 GetNewSize()
	{
		//resize table size to next prime number
		hq_uint32 i = this->m_numBuckets + 1;
		while ( !HQIsPrime(i) || (hq_float32)this->m_numItems / i > 0.5f )
		{
			++i;
		}
		return i;
	}

public:
	HQShaderParamIndexTable() : HQClosedStringHashTable<hq_uint32>(3 , 0.5f) {}
};

#endif
