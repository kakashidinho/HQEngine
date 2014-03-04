/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"

#include <j2cpp/j2cpp.hpp>
#include <android/view/View.hpp>

extern jobject ge_jview;
extern jobject ge_jsurface;
extern jobject ge_jegl;
extern jobject ge_jdisplay;
extern int hq_internalAppExit;

static android::view::View *g_pView;

extern int HQAppInternalGetStatusFlag();
extern void HQEngineInternalRunOnUiThread(void (*runFunc)(void), bool wait);

//helper functions
void ShowWindowFunc()
{
	g_pView->setVisibility(android::view::View::VISIBLE);
}

void HideWindowFunc()
{
	g_pView->setVisibility(android::view::View::INVISIBLE);
}

/*-----------------------engine's window class---------------------------*/

//base class
HQEngineBaseWindow::HQEngineBaseWindow(const char* settingFileDir)
{
	android::view::View viewCpp(ge_jview);

	m_width = viewCpp.getWidth();
	m_height = viewCpp.getHeight();
	
	m_windowed = false;
}

HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* additionalSetting)
: HQEngineBaseWindow(settingFileDir), 
 m_view(ge_jview)
{
	g_pView = &m_view;
	
	//make view visible
	HQEngineInternalRunOnUiThread(&ShowWindowFunc, true);
	
	//create render device init info struct
	if (additionalSetting != NULL)
		m_renderDeviceInitInfo.apiLevel = additionalSetting->openGL_ApiLevel;
	else
		m_renderDeviceInitInfo.apiLevel = 2;
		
	m_renderDeviceInitInfo.jengineView = ge_jview;
	m_renderDeviceInitInfo.jegl = ge_jegl;
	m_renderDeviceInitInfo.jdisplay = ge_jdisplay;
}

HQEngineWindow::~HQEngineWindow()
{
	if (HQAppInternalGetStatusFlag() != hq_internalAppExit)//no need for this if app is exitting
		HQEngineInternalRunOnUiThread(&HideWindowFunc, true);
}

HQRenderDeviceInitInput HQEngineWindow::GetRenderDeviceInitInput() 
{
	return &m_renderDeviceInitInfo;
}

HQReturnVal HQEngineWindow::Show()
{

	return HQ_OK;
}
