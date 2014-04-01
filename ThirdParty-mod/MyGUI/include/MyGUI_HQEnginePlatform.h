/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/

#ifndef __MYGUI_HQEngine_PLATFORM_H__
#define __MYGUI_HQEngine_PLATFORM_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_HQEngineRenderManager.h"
#include "MyGUI_HQEngineDataManager.h"
#include "MyGUI_HQEngineTexture.h"
#include "MyGUI_HQEngineVertexBuffer.h"
#include "MyGUI_HQEngineDiagnostic.h"
#include "MyGUI_LogManager.h"

#include <assert.h>

namespace MyGUI
{

	class HQEnginePlatform
	{
	public:
		HQEnginePlatform() :
			mIsInitialise(false)
		{
			mLogManager = new LogManager();
			mRenderManager = new HQEngineRenderManager();
			mDataManager = new HQEngineDataManager();
		}

		~HQEnginePlatform()
		{
			assert(!mIsInitialise);
			delete mRenderManager;
			delete mDataManager;
			delete mLogManager;
		}

		///
		///Important: HQEngineApp must be initialized
		///
		void initialise(const std::string& _logName = MYGUI_PLATFORM_LOG_FILENAME)
		{
			assert(!mIsInitialise);
			mIsInitialise = true;

			if (!_logName.empty())
				LogManager::getInstance().createDefaultSource(_logName);

			mRenderManager->initialise();
			mDataManager->initialise();
		}

		void shutdown()
		{
			assert(mIsInitialise);
			mIsInitialise = false;

			mRenderManager->shutdown();
			mDataManager->shutdown();
		}

		HQEngineRenderManager* getRenderManagerPtr()
		{
			assert(mIsInitialise);
			return mRenderManager;
		}

		HQEngineDataManager* getDataManagerPtr()
		{
			assert(mIsInitialise);
			return mDataManager;
		}

	private:
		bool                    mIsInitialise;
		HQEngineRenderManager* mRenderManager;
		HQEngineDataManager*   mDataManager;
		LogManager*             mLogManager;

	};

} // namespace MyGUI

#endif // __MYGUI_HQEngine_PLATFORM_H__
