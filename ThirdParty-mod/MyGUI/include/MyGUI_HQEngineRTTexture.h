/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/

#ifndef __MYGUI_HQEngine_RTTEXTURE_H__
#define __MYGUI_HQEngine_RTTEXTURE_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_ITexture.h"
#include "MyGUI_RenderFormat.h"
#include "MyGUI_IRenderTarget.h"

#include "HQRenderDevice.h"

namespace MyGUI
{
	class HQEngineTexture;

	class HQEngineRTTexture :
		public IRenderTarget
	{
	public:
		HQEngineRTTexture( HQEngineTexture* texture, HQEngineRenderManager* manager );
		virtual ~HQEngineRTTexture();

		virtual void begin();
		virtual void end();

		virtual void doRender(IVertexBuffer* _buffer, ITexture* _texture, size_t _count);

		virtual const RenderTargetInfo& getInfo()
		{
			return mRenderTargetInfo;
		}

	private:
		HQViewPort			   mOldViewport;
		HQRenderTargetGroup*   mOldRenderTargetsList;
		HQRenderTargetGroup*   mRenderTargetGroupID;


		HQEngineTexture*       mTexture;
		HQEngineRenderManager* mManager;
		RenderTargetInfo       mRenderTargetInfo;
	};

} // namespace MyGUI

#endif // __MYGUI_DIRECTX_RTTEXTURE_H__
