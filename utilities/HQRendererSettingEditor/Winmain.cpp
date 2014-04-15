#include "resource.h"
#include <d3d9.h>
#include <windows.h>
#include <Commctrl.h>
#pragma comment(lib,"Comctl32.lib")
#include <Commdlg.h>
#pragma comment(lib,"Comdlg32.lib")
#include <stdio.h>
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#include "../../HQEngine/Source/HQPrimitiveDataType.h"

enum FORMAT
{
    SFMT_R8G8B8               = 20,
    SFMT_A8R8G8B8             = 21,
    SFMT_X8R8G8B8             = 22,
    SFMT_R5G6B5               = 23,
    SFMT_X1R5G5B5             = 24,
    SFMT_A1R5G5B5             = 25,
    SFMT_A4R4G4B4             = 26,
    SFMT_R3G3B2               = 27,
    SFMT_A8                   = 28,
    SFMT_A8R3G3B2             = 29,
    SFMT_X4R4G4B4             = 30,
    SFMT_A2B10G10R10          = 31,
    SFMT_A8B8G8R8             = 32,
    SFMT_X8B8G8R8             = 33,
    SFMT_D32                  = 71,
    SFMT_D15S1                = 73,
    SFMT_D24S8                = 75,
    SFMT_D24X8                = 77,
    SFMT_D24X4S4              = 79,
    SFMT_D16                  = 80,
	SFMT_S8					  = 82,
	SFMT_NODEPTHSTENCIL		  = 81
};

const int numBackFmt=9;//số lượng backbuffer format hỗ trợ
const hq_uint32 BackFormat[]={
	SFMT_R8G8B8,   SFMT_A8R8G8B8, 
	SFMT_X8R8G8B8, SFMT_R5G6B5, 
	SFMT_A1R5G5B5, SFMT_X1R5G5B5,
	SFMT_X4R4G4B4, SFMT_A4R4G4B4,
	SFMT_A2B10G10R10 
};

const int numZFmt=8;
const hq_uint32 ZFormat[]={
	SFMT_D24S8,
	SFMT_D24X4S4,
	SFMT_D15S1,
	SFMT_D32,
	SFMT_D24X8,
	SFMT_D16,
	SFMT_S8,
	SFMT_NODEPTHSTENCIL
};

const int numMulSample=4;//số lượng kiểu siêu lấy mẫu hỗ trợ
const D3DMULTISAMPLE_TYPE MulSample[]=
{
	D3DMULTISAMPLE_NONE,
	D3DMULTISAMPLE_2_SAMPLES,
	D3DMULTISAMPLE_4_SAMPLES,
	D3DMULTISAMPLE_8_SAMPLES
};


const int numDevType=2;//số lượng dạng device hỗ trợ
const D3DDEVTYPE DevType[]={
	D3DDEVTYPE_HAL,
	D3DDEVTYPE_REF
};
const DWORD bestVProc=D3DCREATE_HARDWARE_VERTEXPROCESSING|D3DCREATE_PUREDEVICE;
const int numVProc=3;
const DWORD VProc[]={
	bestVProc,
	D3DCREATE_HARDWARE_VERTEXPROCESSING,
	D3DCREATE_SOFTWARE_VERTEXPROCESSING
};
//helper functions
namespace helper{
	//*****************************
	//convert data types to strings
	//*****************************
	void DisplayModeToString(D3DDISPLAYMODE *mode, char *str)
	{
		sprintf(str,"%d X %d",mode->Width,mode->Height);
	}

	void D3DFormatToString(hq_uint32 fmt, char *str)
	{
		switch(fmt)
		{
		case SFMT_R8G8B8: sprintf(str,"R8G8B8");break;
		case SFMT_A8R8G8B8: sprintf(str,"A8R8G8B8");break;
		case SFMT_X8R8G8B8: sprintf(str,"X8R8G8B8");break;
		case SFMT_R5G6B5: sprintf(str,"R5G6B5");break;
		case SFMT_A1R5G5B5: sprintf(str,"A1R5G5B5");break;
		case SFMT_X1R5G5B5: sprintf(str,"X1R5G5B5");break;
		case SFMT_X4R4G4B4: sprintf(str,"X4R4G4B4");break;
		case SFMT_A4R4G4B4: sprintf(str,"A4R4G4B4");break;
		case SFMT_A2B10G10R10: sprintf(str,"A2R10G10B10");break;
		case SFMT_D24S8: sprintf(str,"D24S8");break;
		case SFMT_D24X4S4: sprintf(str,"D24X4S4");break;
		case SFMT_D15S1: sprintf(str,"D15S1");break;
		case SFMT_D32: sprintf(str,"D32");break;
		case SFMT_D24X8: sprintf(str,"D24X8");break;
		case SFMT_D16: sprintf(str,"D16");break;
		case SFMT_S8 : sprintf(str,"S8") ;break;
		case SFMT_NODEPTHSTENCIL : sprintf(str , "No depth stencil buffer"); break;
		default: sprintf(str,"Unknown");break;
		}
	}
	void DevTypeToString(D3DDEVTYPE type,char *str)
	{
		switch(type)
		{
		case D3DDEVTYPE_HAL: sprintf(str,"HAL - fast");break;
		case D3DDEVTYPE_REF: sprintf(str,"REF - slow");break;
		default: sprintf(str,"Unknown");break;
		}
	}


	void MulSampleToString(D3DMULTISAMPLE_TYPE type, char * str)
	{
		switch(type)
		{
		case D3DMULTISAMPLE_NONE: sprintf(str,"None");break;
		case D3DMULTISAMPLE_2_SAMPLES: sprintf(str,"2X");break;
		case D3DMULTISAMPLE_4_SAMPLES: sprintf(str,"4X");break;
		case D3DMULTISAMPLE_8_SAMPLES: sprintf(str,"8X");break;
		default: sprintf(str,"Unknown");break;
		}
	}

	void VerProcBehaviorToString(DWORD behavior,char *str)
	{
		switch(behavior)
		{
		case bestVProc:sprintf(str,"Hardware + Pure Device - fastest");break;
		case D3DCREATE_HARDWARE_VERTEXPROCESSING:sprintf(str,"Hardware - fast");break;
		case D3DCREATE_SOFTWARE_VERTEXPROCESSING:sprintf(str,"Software - medium");break;
		}
	}
	//***************************************
	//combo box control helper functions
	//***************************************
	int AddItemToComboBox(HWND hbox, char *string, void *pData)
	{
		LRESULT index= SendMessageA(hbox,CB_ADDSTRING,0,(LPARAM) string);
		SendMessage(hbox,CB_SETITEMDATA,(WPARAM)index,(LPARAM)pData);
		return (int)index;
	}

	void* GetSelectedItem(HWND hbox)
	{
		LRESULT index=SendMessage(hbox,CB_GETCURSEL,0,0);
		return (void*)SendMessage(hbox,CB_GETITEMDATA,(WPARAM)index,0);
	}
	void SetSelectedItem(HWND hbox,char *string)
	{
		SendMessageA(hbox,CB_SELECTSTRING,(WPARAM)0,(LPARAM)string);
	}
};
INT_PTR CALLBACK SettingDlg(HWND hWndDlg, UINT Msg, WPARAM wParam, LPARAM lParam){
	
	//get dialog items' handles
	HWND hDev=GetDlgItem(hWndDlg,DEV);
	HWND hBfmt=GetDlgItem(hWndDlg,BFMT);
	HWND hZfmt=GetDlgItem(hWndDlg,ZFMT);
	HWND hVproc=GetDlgItem(hWndDlg,VPROC);
	HWND hMulSample=GetDlgItem(hWndDlg,MULSAMPLE);
	HWND hWin=GetDlgItem(hWndDlg,WIN);
	HWND hFull=GetDlgItem(hWndDlg,FULL);
	HWND hVsync=GetDlgItem(hWndDlg,VSYNC);
	HWND hRefresh = GetDlgItem(hWndDlg,REFRESH);
	switch(Msg)
	{
	case WM_INITDIALOG:
		{
			SendMessage(hVsync, BM_SETCHECK, BST_CHECKED, 0);
			SendMessage(hWin, BM_SETCHECK, BST_CHECKED, 0);
			SetDlgItemTextA(hWndDlg,ADAPTER,"0");
			SetDlgItemTextA(hWndDlg,WIDTH,"600");
			SetDlgItemTextA(hWndDlg,HEIGHT,"400");
			SetDlgItemTextA(hRefresh,REFRESH,"60");
			char string[256];
			for(int i=0;i<numDevType;++i)
			{
				helper::DevTypeToString(DevType[i],string);
				helper::AddItemToComboBox(hDev,string,(void*)&DevType[i]);
			}
			SendMessage(hDev,CB_SETCURSEL,0,0);

			for(int i=0;i<numBackFmt;++i)
			{
				helper::D3DFormatToString(BackFormat[i],string);
				helper::AddItemToComboBox(hBfmt,string,(void*)&BackFormat[i]);
			}
			SendMessage(hBfmt,CB_SETCURSEL,0,0);

			for(int i=0;i<numZFmt;++i)
			{
				helper::D3DFormatToString(ZFormat[i],string);
				helper::AddItemToComboBox(hZfmt,string,(void*)&ZFormat[i]);
			}
			SendMessage(hZfmt,CB_SETCURSEL,0,0);

			for(int i=0;i<numVProc;++i)
			{
				helper::VerProcBehaviorToString(VProc[i],string);
				helper::AddItemToComboBox(hVproc,string,(void*)&VProc[i]);
			}
			SendMessage(hVproc,CB_SETCURSEL,0,0);

			for(int i=0;i<numMulSample;++i)
			{
				helper::MulSampleToString(MulSample[i],string);
				helper::AddItemToComboBox(hMulSample,string,(void*)&MulSample[i]);
			}
			SendMessage(hMulSample,CB_SETCURSEL,0,0);
		}
		break;
	case WM_COMMAND:
		switch (LOWORD(wParam))
		{
		case IDOPEN:
			{
				OPENFILENAMEA ofn;
				char szFileName[512] = "";

				ZeroMemory(&ofn, sizeof(ofn));

				ofn.lStructSize = sizeof(ofn); // SEE NOTE BELOW
				ofn.hwndOwner = hWndDlg;
				ofn.lpstrFilter = "All Files (*.*)\0*.*\0";
				ofn.lpstrFile = szFileName;
				ofn.nMaxFile = 512;
				ofn.Flags = OFN_EXPLORER|OFN_FILEMUSTEXIST|OFN_HIDEREADONLY;
				ofn.lpstrDefExt = NULL;
				if(GetOpenFileNameA(&ofn))
				{
					FILE* save;
					save=fopen(ofn.lpstrFile,"r");
					if(!save)
						return FALSE;
					fscanf(save,"Basic Settings\n");
					char n[256];int d;
					fscanf(save,"Width=%s\n",n);
					SetDlgItemTextA(hWndDlg,WIDTH,n);
					fscanf(save,"Height=%s\n",n);
					SetDlgItemTextA(hWndDlg,HEIGHT,n);
					int windowed , vsync;
					fscanf(save,"Windowed=%d\n",&windowed);
					if(windowed)
					{
						SendMessage(hWin, BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(hFull, BM_SETCHECK, BST_UNCHECKED, 0);
					}
					else 
					{
						SendMessage(hFull, BM_SETCHECK, BST_CHECKED, 0);
						SendMessage(hWin, BM_SETCHECK, BST_UNCHECKED, 0);
					}
					fscanf(save,"RefreshRate=%s\n",&n);
					SetDlgItemTextA(hWndDlg,REFRESH,n);
					fscanf(save,"VSync=%d\n\n",&vsync);
					if(vsync)
					{
						SendMessage(hVsync, BM_SETCHECK, BST_CHECKED, 0);
					}
					else
						SendMessage(hVsync, BM_SETCHECK, BST_UNCHECKED, 0);
					fscanf(save,"Advanced Settings\n");
					fscanf(save,"Adapter=%s\n",n);
					SetDlgItemTextA(hWndDlg,ADAPTER,n);
					fscanf(save,"Device Type=%d\n",&d);
					helper::DevTypeToString((D3DDEVTYPE)d,n);
					helper::SetSelectedItem(hDev,n);
					fscanf(save,"BackBuffer Format=%d\n",&d);
					helper::D3DFormatToString(d,n);
					helper::SetSelectedItem(hBfmt,n);
					fscanf(save,"DepthStencil Format=%d\n",&d);
					helper::D3DFormatToString(d,n);
					helper::SetSelectedItem(hZfmt,n);
					fscanf(save,"Vertex Processing=%d\n",&d);
					helper::VerProcBehaviorToString((DWORD)d,n);
					helper::SetSelectedItem(hVproc,n);
					fscanf(save,"Multisample Type=%d\n",&d);
					helper::MulSampleToString((D3DMULTISAMPLE_TYPE)d,n);
					helper::SetSelectedItem(hMulSample,n);
					fclose(save);
				}
			}
			break;
		case IDSAVE:
			{
				OPENFILENAMEA ofn;
				char szFileName[512] = "";

				ZeroMemory(&ofn, sizeof(ofn));

				ofn.lStructSize = sizeof(ofn); // SEE NOTE BELOW
				ofn.hwndOwner = hWndDlg;
				ofn.lpstrFilter = "All Files (*.*)\0*.*\0";
				ofn.lpstrFile = szFileName;
				ofn.nMaxFile = 512;
				ofn.Flags = OFN_EXPLORER|OFN_HIDEREADONLY;
				ofn.lpstrDefExt = NULL;
				if(GetSaveFileNameA(&ofn))
				{
					FILE* save;
					save=fopen(ofn.lpstrFile,"w");
					if(!save)
						return FALSE;
					fprintf(save,"Basic Settings\n");
					char n[128];
				
					GetDlgItemTextA(hWndDlg,WIDTH,n,128);
					int d=atoi(n);
					fprintf(save,"Width=%d\n",d);
					GetDlgItemTextA(hWndDlg,HEIGHT,n,128);
					d=atoi(n);
					fprintf(save,"Height=%d\n",d);
					BOOL windowed=(SendMessage(hWin,BM_GETCHECK,0,0)==BST_CHECKED);
					fprintf(save,"Windowed=%d\n",windowed);
					GetDlgItemTextA(hWndDlg,REFRESH,n,128);
					d=atoi(n);
					fprintf(save,"RefreshRate=%d\n",d);
					BOOL vsync=(SendMessage(hVsync,BM_GETCHECK,0,0)==BST_CHECKED);
					fprintf(save,"VSync=%d\n\n",vsync);
					
					fprintf(save,"Advanced Settings\n");
					GetDlgItemTextA(hWndDlg,ADAPTER,n,128);
					d=atoi(n);
					fprintf(save,"Adapter=%d\n",d);
					D3DDEVTYPE* dev=(D3DDEVTYPE*)helper::GetSelectedItem(hDev);
					fprintf(save,"Device Type=%d\n",(int)*dev);
					FORMAT* Bfmt=(FORMAT*)helper::GetSelectedItem(hBfmt);
					fprintf(save,"BackBuffer Format=%d\n",(int)*Bfmt);
					FORMAT* Zfmt=(FORMAT*)helper::GetSelectedItem(hZfmt);
					fprintf(save,"DepthStencil Format=%d\n",(int)*Zfmt);
					DWORD* behavior=(DWORD*)helper::GetSelectedItem(hVproc);
					fprintf(save,"Vertex Processing=%d\n",(int)*behavior);
					D3DMULTISAMPLE_TYPE* mulsample=(D3DMULTISAMPLE_TYPE*)helper::GetSelectedItem(hMulSample);
					fprintf(save,"Multisample Type=%d\n",(int)*mulsample);
					fclose(save);
				}
			}
			break;
		case IDCANCEL:
			EndDialog(hWndDlg,0);
			return TRUE;
			break;
		}
		break;
	}
	return 0; 
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	InitCommonControls();
	DialogBoxA(hInstance,"Setting",0,SettingDlg);
	return 0;
}