/*!
	@file
	@author		Albert Semenov
	@date		04/2009
*/
/*
	This file is part of MyGUI.

	MyGUI is free software: you can redistribute it and/or modify
	it under the terms of the GNU Lesser General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	MyGUI is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with MyGUI.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __MYGUI_I_TEXTURE_H__
#define __MYGUI_I_TEXTURE_H__

#include "MyGUI_Prerequest.h"
#include "MyGUI_Types.h"
#include "MyGUI_IRenderTarget.h"
#include "MyGUI_RenderFormat.h"
#include <string>

namespace MyGUI
{
	class ITexture;

	class MYGUI_EXPORT ITextureInvalidateListener
	{
	public:
		virtual ~ITextureInvalidateListener() { }
		virtual void textureInvalidate(ITexture* _texture) = 0;
	};


	class MYGUI_EXPORT ITexturePixelBuffer
	{
	public:
		virtual ~ITexturePixelBuffer(){}
		///
		///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
		///A is ignored in R5G6B5 buffer. Color channel range is 0.0f-1.0f
		///
		virtual void setPixelf(uint32 x, uint32 y, float r, float g, float b, float a) = 0;
		///
		///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
		///A is ignored in R5G6B5 buffer. Color channel range is 0-255
		///
		virtual void setPixel(uint32 x, uint32 y, uint8 r, uint8 g, uint8 b, uint8 a) = 0;

		virtual void setPixelf(uint32 x, uint32 y, float r, float g, float b) = 0;///alpha is assumed to be 1.0f. Same as SetPixel(x, y, r, g, b, 1)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 r, uint8 g, uint8 b) = 0;///alpha is assumed to be 255. Same as SetPixel(x, y, r, g, b, 1)
		
		virtual void setPixelf(uint32 x, uint32 y, float l, float a) = 0;///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 l, uint8 a) = 0;///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
		
		virtual void setPixelf(uint32 x, uint32 y, float a) = 0;///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)
		
		virtual void setPixel(uint32 x, uint32 y, uint8 a) = 0;///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)

		virtual uint32 getPixel(uint32 x, uint32 y) = 0;//returned color is in ABGR order where A is in most significant bits
	};

	class MYGUI_EXPORT ITexture
	{
	public:
		virtual ~ITexture() { }

		virtual const std::string& getName() const = 0;

		virtual void createManual(int _width, int _height, TextureUsage _usage, PixelFormat _format) = 0;
		virtual void loadFromFile(const std::string& _filename) = 0;
		virtual void saveToFile(const std::string& _filename) = 0;

		virtual void setInvalidateListener(ITextureInvalidateListener* _listener) { }

		virtual void destroy() = 0;

		virtual ITexturePixelBuffer* lockPixelBuffer(TextureUsage _usage) = 0;
		virtual void unlockPixelBuffer() = 0;
		virtual bool isLockedPixelBuffer() = 0;

		virtual int getWidth() = 0;
		virtual int getHeight() = 0;

		virtual PixelFormat getFormat() = 0;
		virtual TextureUsage getUsage() = 0;

		virtual IRenderTarget* getRenderTarget()
		{
			return nullptr;
		}
	};

} // namespace MyGUI

#endif // __MYGUI_I_TEXTURE_H__
