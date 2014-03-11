using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace HQEngineDeployDll
{

    class FileInfo
    {
        public FileInfo(string name)
        {
            this._name = name;
            this._base = false;
        }
        public FileInfo( bool isBase, string name)
        {
            this._name = name;
            this._base = isBase;
        }
        public string name
        {
            get { return _name; }
        }
        public bool isBase
        {
            get { return _base; }
        }

        public static implicit operator FileInfo(string name)
        {
            return new FileInfo(name);
        }

        private string _name;
        private bool _base;
    };

    static class Global{
        /*------------D3D9--------------*/
        public static FileInfo[] D3D9_Dlls = {
	        new FileInfo(true,"HQRendererD3D9.dll"),
	        new FileInfo(false, "cg.dll"),
	        new FileInfo(false,"cgD3D9.dll"),
	        new FileInfo(false,"D3DX9_43.dll")
        };

        public static FileInfo[] D3D9D_Dlls = {
	        new FileInfo(true,"HQRendererD3D9_D.dll"),
	        "cg.dll",
	        "cgD3D9.dll",
	        "D3DX9d_43.dll",
	        "d3d9d.dll",
	        "d3dref9.dll",
	        "PVRTexLib.dll"
        };

        /*------------D3D11--------------*/
        public static FileInfo[] D3D11_Dlls = {
	        new FileInfo(true,"HQRendererD3D11.dll"),
	        "cg.dll",
	        "cgD3D11.dll",
	        "D3DX11_43.dll"
        };

        public static FileInfo[] D3D11D_Dlls = {
	        new FileInfo(true,"HQRendererD3D11_D.dll"),
	        "cg.dll",
	        "cgD3D11.dll",
	        "D3DX11d_43.dll",
	        "D3D11SDKLayers.dll",
	        "d3d11ref.dll",
	        "PVRTexLib.dll"
        };

        /*------------GL--------------*/
        public static FileInfo[] GL_Dlls = {
	        new FileInfo(true,"HQRendererGL.dll")
        };

        public static FileInfo[] GLD_Dlls = {
	        new FileInfo(true,"HQRendererGL_D.dll"),
	        "PVRTexLib.dll"
        };


        /*------------GL with cg--------------*/
        public static FileInfo[] GL_CG_Dlls = {
	        new FileInfo(true,"HQRendererGL.dll"),
	        "cg.dll",
	        "cgGL.dll"
        };


        public static FileInfo[] GLD_CG_Dlls = {
	        new FileInfo(true,"HQRendererGL_D.dll"),
	        "cg.dll",
	        "cgGL.dll",
	        "PVRTexLib.dll"
        };

        /*------------Audio XAudio2--------------*/
        public static FileInfo[] XAudio_Dlls = {
	        new FileInfo(true,"HQAudio.dll"),
	        "X3DAudio1_7.dll",
	        "XAudio2_7.dll"
        };

        public static FileInfo[] XAudioD_Dlls = {
	        new FileInfo(true,"HQAudioD.dll"),
	        "X3DAudioD1_7.dll",
	        "XAudioD2_7.dll"
        };

        /*------------Audio with openAL--------------*/
        public static FileInfo[] AudioAL_Dlls = {
	        new FileInfo(true,"HQAudio.dll"),
	        "wrap_oal.dll",
	        "OpenAL32.dll"
        };


        public static FileInfo[] AudioALD_Dlls = {
	        new FileInfo(true,"HQAudioD.dll"),
	        "wrap_oal.dll",
	        "OpenAL32.dll"
        };

        /*---------3d math-----------*/
        public static FileInfo[] _3DMath_Dlls = {
	        new FileInfo(true,"HQUtilMath.dll")
        };


        public static FileInfo[] _3DMathD_Dlls = {
	        new FileInfo(true,"HQUtilMathD.dll")
        };


        /*---------util-----------*/
        public static FileInfo[] Util_Dlls = {
	        new FileInfo(true,"HQUtil.dll")
        };

        public static FileInfo[] UtilD_Dlls = {
	        new FileInfo(true,"HQUtilD.dll")
        };


        /*---------engine-----------*/
        public static FileInfo[] Engine_Dlls = {
	        new FileInfo(true,"HQEngine.dll"),
	        new FileInfo(true,"HQUtil.dll")
        };


        public static FileInfo[] EngineD_Dlls = {
	        new FileInfo(true,"HQEngineD.dll"),
	        new FileInfo(true,"HQUtilD.dll")
        };

        public static string DebugBaseDllPath = "";
        public static string ReleaseBaseDllPath = "";
        public static string SrcDllPath = "";
        public static string DestPath = "";
        public static string currentWorkingDir;

        public static bool D3D9 = false;
        public static bool D3D11 = false;
        public static bool GL = false;
        public static bool cg = false;
        public static bool Audio = false;
        public static bool openAL = false;
        public static bool Math = false;
        public static bool Util = false;
        public static bool Engine = false;
        public static bool Debug = false;
    }
    static class Program
    {
        private static Form1 form;
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Global.currentWorkingDir = System.IO.Directory.GetCurrentDirectory();
            

            try
            {
                System.IO.StreamReader stream = new System.IO.StreamReader("setting.ini");
                string line;
                if ((line = stream.ReadLine()) != null)
                    Global.DebugBaseDllPath = line.Substring(line.IndexOf('=')+1);
                if ((line = stream.ReadLine()) != null)
                    Global.ReleaseBaseDllPath = line.Substring(line.IndexOf('=') + 1);
                if ((line = stream.ReadLine()) != null)
                    Global.SrcDllPath = line.Substring(line.IndexOf('=') + 1);
                if ((line = stream.ReadLine()) != null)
                    Global.DestPath = line.Substring(line.IndexOf('=') + 1);

                stream.Close();
            }
            catch (Exception e)
            {

            }

            CorrectPaths();

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(form = new Form1());

            SaveSetting();
        }

        private static void CorrectPaths()//add "\\" to the end if it doesn't exists
        {
            CorrectPath(ref Global.currentWorkingDir);

            CorrectPath(ref Global.DebugBaseDllPath);

            CorrectPath(ref Global.ReleaseBaseDllPath);

            CorrectPath(ref Global.SrcDllPath);

            CorrectPath(ref Global.DestPath);
        }

        public static void CorrectPath(ref string path)
        {
            if (path.Length > 0 && path[path.Length - 1] != '\\')
                path += "\\";
        }

        public static void SaveSetting()
        {
            System.IO.StreamWriter writter = new System.IO.StreamWriter("setting.ini");
            writter.WriteLine("Debug=" + Global.DebugBaseDllPath);
            writter.WriteLine("Release=" + Global.ReleaseBaseDllPath);
            writter.WriteLine("Source=" + Global.SrcDllPath);
            writter.WriteLine("Dest=" + Global.DestPath);
            writter.Close();
        }

        public static void Remove()
        {
            try
            {
                Process(RemoveDlls);
            }
            catch (Exception e)
            {
                System.IO.Directory.SetCurrentDirectory(Global.currentWorkingDir);
            }
        }
        public static void Deploy()
        {
            try
            {
                Process(DeployDlls);
            }
            catch (Exception e)
            {
                System.IO.Directory.SetCurrentDirectory(Global.currentWorkingDir);
            }
        }

        private static void RemoveDlls(FileInfo[] dlls, ref string log)
        {
            System.IO.DirectoryInfo destDir = null;
            
            if (!System.IO.Directory.Exists(Global.DestPath))
            {
                MessageBox.Show(form, "Deployment directory not found!", "Error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                throw new System.IO.DirectoryNotFoundException();
            }
            else
            {
                destDir = System.IO.Directory.CreateDirectory(Global.DestPath);
                System.IO.Directory.SetCurrentDirectory(destDir.FullName);
            }

            for (int i = 0; i < dlls.Length; ++i)
            {
                if (System.IO.File.Exists(dlls[i].name))
                {
                    System.IO.File.Delete(dlls[i].name);
                    log += destDir.FullName + dlls[i].name + " removed" + Environment.NewLine + Environment.NewLine;
                }
            }

            System.IO.Directory.SetCurrentDirectory(Global.currentWorkingDir);
        }

        private static void DeployDlls(FileInfo[] dlls, ref string log)
        {
            System.IO.DirectoryInfo baseDir = null;
            System.IO.DirectoryInfo srcDir = null;
            System.IO.DirectoryInfo destDir = null;
            
            bool existBaseDllPath = true;
            
            if (Global.Debug)
            {
                if (!System.IO.Directory.Exists(Global.DebugBaseDllPath))
                    existBaseDllPath = false;
                else
                    baseDir = System.IO.Directory.CreateDirectory(Global.DebugBaseDllPath);
            }
            else
            {
                if (!System.IO.Directory.Exists(Global.ReleaseBaseDllPath))
                    existBaseDllPath = false;
                else
                    baseDir = System.IO.Directory.CreateDirectory(Global.ReleaseBaseDllPath);
            }

            if (!existBaseDllPath)
            {
                MessageBox.Show(form, "Base Directory not found!", "Error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                throw new System.IO.DirectoryNotFoundException() ;
            }

            if (System.IO.Directory.Exists(Global.SrcDllPath))
            {
                srcDir = System.IO.Directory.CreateDirectory(Global.SrcDllPath);
            }
            

            if (!System.IO.Directory.Exists(Global.DestPath))
            {
                MessageBox.Show(form, "Deployment directory not found!", "Error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                throw new System.IO.DirectoryNotFoundException();
            }
            else
            {
                destDir = System.IO.Directory.CreateDirectory(Global.DestPath);
                System.IO.Directory.SetCurrentDirectory(destDir.FullName);
            }

            string fullFilePath = null;


            for (int i = 0; i < dlls.Length; ++i)
            {
                if (dlls[i].isBase)
                    fullFilePath = baseDir.FullName + dlls[i].name;
                else if (srcDir == null)
                {
                    MessageBox.Show(form, "File " + dlls[i].name + " not found!", "Error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    continue;
                }
                else
                    fullFilePath = srcDir.FullName + dlls[i].name;

                if (!System.IO.File.Exists(fullFilePath))
                {
                    MessageBox.Show(form, "File " + dlls[i].name + " not found!", "Error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    continue;
                }

                if (!System.IO.File.Exists(dlls[i].name))
                {
                    System.IO.File.Copy(fullFilePath, dlls[i].name);
                    log += destDir.FullName + dlls[i].name + " copied" + Environment.NewLine + Environment.NewLine;
                }
                else
                    log += destDir.FullName + dlls[i].name + " already exists" + Environment.NewLine + Environment.NewLine;
            }

            System.IO.Directory.SetCurrentDirectory(Global.currentWorkingDir);
        }

        private static void Process(ProcessDllsDelegate process)
        {
            string log = "";
            if (Global.Debug)
            {
                if (Global.D3D9)
                {
                    process(Global.D3D9D_Dlls, ref log);
                }
                if (Global.D3D11)
                {
                    process(Global.D3D11D_Dlls, ref log);
                }
                if (Global.GL)
                {
                    if (Global.cg)
                        process(Global.GLD_CG_Dlls, ref log);
                    else
                        process(Global.GLD_Dlls, ref log);
                }
                if (Global.Audio)
                {
                    if (Global.openAL)
                        process(Global.AudioALD_Dlls, ref log);
                    else
                        process(Global.XAudioD_Dlls, ref log);
                }
                if (Global.Math)
                    process(Global._3DMathD_Dlls, ref log);
                if (Global.Util)
                    process(Global.UtilD_Dlls, ref log);
                if (Global.Engine)
                    process(Global.EngineD_Dlls, ref log);
            }
            else//release
            {
                if (Global.D3D9)
                {
                    process(Global.D3D9_Dlls, ref log);
                }
                if (Global.D3D11)
                {
                    process(Global.D3D11_Dlls, ref log);
                }
                if (Global.GL)
                {
                    if (Global.cg)
                        process(Global.GL_CG_Dlls, ref log);
                    else
                        process(Global.GL_Dlls, ref log);
                }
                if (Global.Audio)
                {
                    if (Global.openAL)
                        process(Global.AudioAL_Dlls, ref log);
                    else
                        process(Global.XAudio_Dlls, ref log);
                }
                if (Global.Math)
                    process(Global._3DMath_Dlls, ref log);
                if (Global.Util)
                    process(Global.Util_Dlls, ref log);
                if (Global.Engine)
                    process(Global.Engine_Dlls, ref log);
            }


            form.SetLog(log);
        }

        private delegate void ProcessDllsDelegate(FileInfo[] dlls, ref string log);
    }
}
