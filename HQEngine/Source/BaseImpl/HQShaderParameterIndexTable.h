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
