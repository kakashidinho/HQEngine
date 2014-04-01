/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/
#ifndef __MYGUI_HQEngine_RENDER_MANAGER_H__
#define __MYGUI_HQEngine_RENDER_MANAGER_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_RenderFormat.h"
#include "MyGUI_IVertexBuffer.h"
#include "MyGUI_RenderManager.h"

#include "HQRendererCoreType.h"

class HQRenderDevice;

namespace MyGUI
{
	class HQEngineTexture;

	class HQEngineRenderManager :
		public RenderManager,
		public IRenderTarget
	{
	public:
		HQEngineRenderManager();

		///
		///Important: HQEngineApp must be initialized
		///
		void initialise();
		void shutdown();

		static HQEngineRenderManager& getInstance()
		{
			return *getInstancePtr();
		}
		static HQEngineRenderManager* getInstancePtr()
		{
			return static_cast<HQEngineRenderManager*>(RenderManager::getInstancePtr());
		}

		/** @see RenderManager::getViewSize */
		virtual const IntSize& getViewSize() const
		{
			return mViewSize;
		}

		/** @see RenderManager::getVertexFormat */
		virtual VertexColourType getVertexFormat()
		{
			return mVertexFormat;
		}

		/** @see RenderManager::createVertexBuffer */
		virtual IVertexBuffer* createVertexBuffer();
		/** @see RenderManager::destroyVertexBuffer */
		virtual void destroyVertexBuffer(IVertexBuffer* _buffer);

		/** @see RenderManager::createTexture */
		virtual ITexture* createTexture(const std::string& _name);
		/** @see RenderManager::destroyTexture */
		virtual void destroyTexture(ITexture* _texture);
		/** @see RenderManager::getTexture */
		virtual ITexture* getTexture(const std::string& _name);

		/** @see RenderManager::isFormatSupported */
		virtual bool isFormatSupported(PixelFormat _format, TextureUsage _usage);

		/** @see IRenderTarget::begin */
		virtual void begin();
		/** @see IRenderTarget::end */
		virtual void end();

		/** @see IRenderTarget::doRender */
		virtual void doRender(IVertexBuffer* _buffer, ITexture* _texture, size_t _count);

		/** @see IRenderTarget::getInfo */
		virtual const RenderTargetInfo& getInfo()
		{
			return mInfo;
		}

		/*internal:*/
		void drawOneFrame();
		void setViewSize(int _width, int _height);

		bool isOpenGL() const {return mIsOpenGL;}

		void onDeviceLost();
		void onDeviceReset();

	private:
		void destroyAllResources();

	public:
		HQRenderDevice*			mpDevice;
		hquint32				mBlendState;
		hquint32				mDepthStencilState;
		hquint32				mSamplerState;
		HQVertexLayout*			mInputLayout;

	private:
		IntSize                  mViewSize;
		VertexColourType         mVertexFormat;
		RenderTargetInfo         mInfo;
		bool					 mIsOpenGL;
		bool                     mUpdate;

		typedef std::map<std::string, HQEngineTexture*> MapTexture;

		MapTexture           mTextures;
		bool                 mIsInitialise;
	};

} // namespace MyGUI

#endif // __MYGUI_HQEngine_RENDER_MANAGER_H__
