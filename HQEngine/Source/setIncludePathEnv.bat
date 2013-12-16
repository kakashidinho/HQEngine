set CDIR=%~dp0

reg add HKCU\Environment  /v HQENGINE_INC_PATH /t REG_SZ /d %CDIR%