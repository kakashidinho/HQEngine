set CDIR=%~dp0

reg add HKCU\Environment  /v HQENGINE_VS2008_X86_LIB_DEB_PATH /t REG_SZ /d %CDIR%\Output\Debug\
reg add HKCU\Environment  /v HQENGINE_VS2008_X86_LIB_REL_PATH /t REG_SZ /d %CDIR%\Output\Release\
reg add HKCU\Environment  /v HQENGINE_VS2008_X86_LIB_STATIC_CRT_DEB_PATH /t REG_SZ /d "%CDIR%\Output\Debug static CRT\"
reg add HKCU\Environment  /v HQENGINE_VS2008_X86_LIB_STATIC_CRT_REL_PATH /t REG_SZ /d "%CDIR%\Output\Release static CRT\"