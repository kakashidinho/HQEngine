#include "Utils.h"

#include <stdio.h>
#include <string.h>

bool WriteToBMP(const char *fileName, const void *data, unsigned int width, unsigned int height, int bits, bool flipRGB)
{
	//dump the height data to greyscale format
	FILE * f = fopen(fileName, "wb");
	if (f == NULL)
		return false;
	unsigned int pixelSize = bits / 8;
	//write header
	unsigned int rowSize = width * pixelSize;
	int linePadding = (4 - rowSize % 4) % 4;
	unsigned int imgSize = (rowSize + linePadding) * height;
	unsigned int fileSize = 54 + imgSize;
	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, bits, 0,
		0, 0, 0, 0, // compression is none
		0, 0, 0, 0, // image size
		0x13, 0x0B, 0, 0, // horz resoluition in pixel / m
		0x13, 0x0B, 0, 0 // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
	};
	unsigned char bmppad[3] = { 0, 0, 0 };

	memcpy(bmpfileheader + 2, &fileSize, 4);//file size
	memcpy(bmpinfoheader + 4, &width, 4);//image width
	memcpy(bmpinfoheader + 8, &height, 4);//image height
	memcpy(bmpinfoheader + 24, &imgSize, 4);//image size

	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);


	//write pixel data
	const unsigned char* pPixelData = (const unsigned char*)data;
	for (unsigned int i = 0; i < height; i++)
	{
		for (unsigned int j = 0; j < width; j++)
		{
			unsigned char pixelChannel[4];

			if (bits >= 24 && flipRGB)
			{
				pixelChannel[0] = pPixelData[2];
				pixelChannel[1] = pPixelData[1];
				pixelChannel[2] = pPixelData[0];
				if (pixelSize == 4)
					pixelChannel[3] = pPixelData[3];
			}
			else
			{
				for (unsigned int i = 0; i < pixelSize; ++i)
				{
					pixelChannel[i] = pPixelData[i];
				}
			}
			
			fwrite(pixelChannel, pixelSize, 1, f);
			
			pPixelData += pixelSize;
		}
	
		fwrite(bmppad, 1, linePadding, f);//line padding

	}

	fclose(f);

	return true;
}