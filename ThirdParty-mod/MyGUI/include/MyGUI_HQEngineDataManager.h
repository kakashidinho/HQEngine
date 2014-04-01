/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/
#ifndef __MYGUI_HQEngine_DATA_MANAGER_H__
#define __MYGUI_HQEngine_DATA_MANAGER_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_DataManager.h"

namespace MyGUI
{

	class HQEngineDataManager :
		public DataManager
	{
	public:
		HQEngineDataManager();

		void initialise();
		void shutdown();

		static HQEngineDataManager& getInstance()
		{
			return *getInstancePtr();
		}
		static HQEngineDataManager* getInstancePtr()
		{
			return static_cast<HQEngineDataManager*>(DataManager::getInstancePtr());
		}

		/** @see DataManager::getData(const std::string& _name) */
		virtual IDataStream* getData(const std::string& _name);

		/** @see DataManager::isDataExist(const std::string& _name) */
		virtual bool isDataExist(const std::string& _name);

		/** @see DataManager::getDataListNames(const std::string& _pattern) */
		virtual const VectorString& getDataListNames(const std::string& _pattern);

		/** @see DataManager::getDataPath(const std::string& _name) */
		virtual const std::string& getDataPath(const std::string& _name);

		/*internal:*/
		void addResourceLocation(const std::string& _name, bool _recursive);

	private:
		struct ArhivInfo
		{
#if MYGUI_IS_WSTRING_SUPPORT
			std::wstring name;
#else
			std::string name;
#endif
			bool recursive;
		};
		typedef std::vector<ArhivInfo> VectorArhivInfo;
		VectorArhivInfo mPaths;

		bool mIsInitialise;
	};

} // namespace MyGUI

#endif // __MYGUI_HQEngine_DATA_MANAGER_H__
