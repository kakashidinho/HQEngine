/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/

#ifndef __MYGUI_HQEngine_VERTEX_BUFFER_H__
#define __MYGUI_HQEngine_VERTEX_BUFFER_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_IVertexBuffer.h"
#include "MyGUI_HQEngineRenderManager.h"

#include "HQPrimitiveDataType.h"

namespace MyGUI
{

	class HQEngineVertexBuffer : public IVertexBuffer
	{
	public:
		HQEngineVertexBuffer(HQEngineRenderManager* _pRenderManager);
		virtual ~HQEngineVertexBuffer();

		virtual void setVertexCount(size_t _count);
		virtual size_t getVertexCount();

		virtual Vertex* lock();
		virtual void unlock();

	private:
		bool create();
		void destroy();
		void resize();

	private:
		HQEngineRenderManager*	mManager;
		size_t                  mVertexCount;
		size_t                  mNeedVertexCount;

	public:
		HQVertexBuffer*         mBuffer;
	};

} // namespace MyGUI

#endif // __MYGUI_HQEngine_VERTEX_BUFFER_H__
