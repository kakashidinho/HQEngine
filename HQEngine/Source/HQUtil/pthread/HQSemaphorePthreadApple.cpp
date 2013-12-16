#include "../HQUtilPCH.h"
#include "../../HQSemaphore.h"
#include "../../HQMutex.h"
#include <semaphore.h>

static char nameBuffer[12];
static unsigned int num = 0;
static HQMutex mutex;


struct HQSemInfo
{
	HQSemInfo(sem_t * pSemID)
	:m_pSemID (pSemID)
	{
		m_name = new char [strlen(nameBuffer) + 1];
		strcpy(m_name, nameBuffer);
	}
	~HQSemInfo()
	{
		delete[] m_name;
	}
	sem_t * m_pSemID;
	char * m_name;
};

HQSemaphore::HQSemaphore(hq_int32 initValue)

{
	if (initValue < 1)
		initValue = 1;
	
	HQMutex::ScopeLock sl(mutex);
	
	
	
	sem_t* pSemID;
	bool success = false;
	mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
	unsigned int numTry = 0;//number of try
	
	do{
		++numTry;
		sprintf(nameBuffer, "/%u" , num);//create unique name
	
		pSemID = sem_open(nameBuffer, O_CREAT | O_EXCL , mode , (unsigned int) initValue );//create named semaphore
	
		if(pSemID == SEM_FAILED )
		{
			if (EEXIST == errno)//exists
			{
				if (num == ~(unsigned int)0)//max value
					num = 0;
				else
					++num;//next id
			}
			else
				throw std::bad_alloc();
		}
		else
			success = true;
	} while ( !success && numTry < (~(unsigned int)0)) ;
	
	
	
	try{
		m_platformSpecific = new HQSemInfo(pSemID);
	}
	catch ( ... ) {
		sem_close(pSemID);
		sem_unlink(nameBuffer);
		
		throw std::bad_alloc();
	}
	
}

HQSemaphore::~HQSemaphore()
{
	HQSemInfo *info = (HQSemInfo*) m_platformSpecific;
	sem_close(info->m_pSemID);
	sem_unlink(info->m_name);
	delete ((HQSemInfo*) m_platformSpecific);
}

void HQSemaphore::Lock()
{
	sem_wait(((HQSemInfo*) m_platformSpecific)->m_pSemID);
}

void HQSemaphore::Unlock()
{
	sem_post(((HQSemInfo*) m_platformSpecific)->m_pSemID);
}