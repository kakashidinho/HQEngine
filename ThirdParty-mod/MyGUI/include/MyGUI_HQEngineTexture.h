/*!
	@file
	@author		Ustinov Igor aka Igor', DadyaIgor
	@date		09/2011
*/

#ifndef __MYGUI_HQEngine_TEXTURE_H__
#define __MYGUI_HQEngine_TEXTURE_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_ITexture.h"
#include "MyGUI_RenderFormat.h"
#include "MyGUI_Types.h"

#include "HQRendererCoreType.h"

class HQRawPixelBuffer;
class HQRenderDevice;

namespace MyGUI
{
	class HQEngineRenderManager;
	class HQEngineRTTexture;

	class HQEngineTexturePixelBuffer: public ITexturePixelBuffer
	{
	public:
		HQEngineTexturePixelBuffer(HQRenderDevice *renderDevice, uint32 width, uint32 height, PixelFormat format);
		~HQEngineTexturePixelBuffer();
		///
		///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
		///A is ignored in R5G6B5 buffer. Color channel range is 0.0f-1.0f
		///
		virtual void setPixelf(uint32 x, uint32 y, float r, float g, float b, float a) ;
		///
		///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
		///A is ignored in R5G6B5 buffer. Color channel range is 0-255
		///
		virtual void setPixel(uint32 x, uint32 y, uint8 r, uint8 g, uint8 b, uint8 a) ;

		virtual void setPixelf(uint32 x, uint32 y, float r, float g, float b) ;///alpha is assumed to be 1.0f. Same as SetPixel(x, y, r, g, b, 1)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 r, uint8 g, uint8 b) ;///alpha is assumed to be 255. Same as SetPixel(x, y, r, g, b, 1)
		
		virtual void setPixelf(uint32 x, uint32 y, float l, float a) ;///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 l, uint8 a) ;///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
		
		virtual void setPixelf(uint32 x, uint32 y, float a) ;///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 a) ;///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)

		virtual uint32 getPixel(uint32 x, uint32 y) ;//returned color is in ABGR order where A is in most significant bits
		
		const HQRawPixelBuffer * getHQEngineBuffer() const {return mBuffer;}

	private:
		HQRawPixelBuffer * mBuffer;
	};

	class HQEngineTexture : public ITexture
	{
	public:
		HQEngineTexture(const std::string& _name, HQEngineRenderManager* _manager);
		virtual ~HQEngineTexture();

		virtual const std::string& getName() const;

		virtual void createManual(int _width, int _height, TextureUsage _usage, PixelFormat _format);
		virtual void loadFromFile(const std::string& _filename);
		virtual void saveToFile(const std::string& _filename) { }

		virtual void destroy();

		virtual ITexturePixelBuffer* lockPixelBuffer(TextureUsage _usage) ;
		virtual void unlockPixelBuffer();
		virtual bool isLockedPixelBuffer();

		virtual int getWidth();
		virtual int getHeight();

		virtual PixelFormat  getFormat();
		virtual TextureUsage getUsage();

		virtual IRenderTarget* getRenderTarget();

		virtual void setInvalidateListener(ITextureInvalidateListener* _listener) {mInvalidateListener = _listener;}
		
		void onDeviceLost();
		void onDeviceReset();

		HQTexture*			  getRawTexture() const {return mTexture;}
		HQRenderTargetView*	  getRawRenderTarget() const {return mRenderTargetID;}
		
		void bindSamplerState(hquint32 stateID) {mSamplerStateID = stateID;}///this is useful in openGL renderer
	private:
		friend class					HQEngineRTTexture;
		HQTexture*						mTexture;
		HQEngineTexturePixelBuffer*		mWriteDataBuffer;

		hquint32						mSamplerStateID;

		ITextureInvalidateListener*		mInvalidateListener;

		int								mWidth, mHeight;
		TextureUsage					mTextureUsage;
		PixelFormat						mFormat;
		std::string						mName;
		std::string						mFileName;
		bool							mLock;
		HQRenderTargetView*				mRenderTargetID;
		HQEngineRTTexture*				mRenderTarget;
		HQEngineRenderManager*			mManager;
	};

} // namespace MyGUI

#endif // __MYGUI_HQEngine_TEXTURE_H__
