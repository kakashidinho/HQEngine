using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace HQEngineDeployDll
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            System.IO.DirectoryInfo info = null;

            if (!System.IO.Directory.Exists(Global.SrcDllPath))
                Global.SrcDllPath = this.srcPathBox.Text = Global.currentWorkingDir;
            else
            {


                info = System.IO.Directory.CreateDirectory(Global.SrcDllPath);

                string path = info.FullName;

                Program.CorrectPath(ref path);

                this.srcPathBox.Text = path;
            }

            if (!System.IO.Directory.Exists(Global.DestPath))
                Global.DestPath = this.destPathBox.Text = Global.currentWorkingDir;
            else
            {
                info = System.IO.Directory.CreateDirectory(Global.DestPath);

                string path = info.FullName;

                Program.CorrectPath(ref path);

                this.destPathBox.Text = path;
            }
            
            

            this.GL.CheckedChanged += new EventHandler(GL_CheckedChanged);
            this.Audio.CheckedChanged += new EventHandler(Audio_CheckedChanged);

            //this.folderBrowserDialog.RootFolder = Environment.SpecialFolder.Desktop;
            
        }

        void Audio_CheckedChanged(object sender, EventArgs e)
        {
            CheckBox Audio = (CheckBox)sender;
            if (Audio.Checked)
            {
                this.XAudio.Enabled = true;
                this.openAL.Enabled = true;
            }
            else
            {
                this.XAudio.Enabled = false;
                this.openAL.Enabled = false;
            }
        }

        void GL_CheckedChanged(object sender, EventArgs e)
        {
            CheckBox GL = (CheckBox) sender;
            if (GL.Checked)
                this.cg.Enabled = true;
            else
                this.cg.Enabled = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //Program.SaveSetting();
            this.Close();
        }

        private void GetOptions()
        {
            Global.D3D9 = this.D3D9.Checked;
            Global.D3D11 = this.D3D11.Checked;
            Global.GL = this.GL.Checked;
            Global.cg = this.cg.Checked;
            Global.Audio = this.Audio.Checked;
            Global.openAL = this.openAL.Checked;
            Global.Math = this.Math.Checked;
            Global.Util = this.Util.Checked;
            Global.Engine = this.Engine.Checked;
            Global.Debug = this.Debug.Checked;
        }

        private void button4_Click(object sender, EventArgs e)//remove
        {
            GetOptions();

            Program.Remove();
        }

        private void button5_Click(object sender, EventArgs e)//deploy
        {
            GetOptions();

            Program.Deploy();
        }

        private void button2_Click(object sender, EventArgs e)//select source folder
        {
            if (System.IO.Directory.Exists(Global.SrcDllPath))
            {
                System.IO.DirectoryInfo info = System.IO.Directory.CreateDirectory(Global.SrcDllPath);
                folderBrowserDialog.SelectedPath = info.FullName;
            }
            

            DialogResult result = folderBrowserDialog.ShowDialog(this);
            if (result == DialogResult.OK)
            {
                string path = folderBrowserDialog.SelectedPath;

                Program.CorrectPath(ref path);

                Global.SrcDllPath = this.srcPathBox.Text = path;
            }
        }

        private void button3_Click(object sender, EventArgs e)//select dest folder
        {
            if (System.IO.Directory.Exists(Global.DestPath))
            {
                System.IO.DirectoryInfo info = System.IO.Directory.CreateDirectory(Global.DestPath);
                folderBrowserDialog.SelectedPath = info.FullName;
            }
            
            DialogResult result = folderBrowserDialog.ShowDialog(this);
            if (result == DialogResult.OK)
            {
                string path = folderBrowserDialog.SelectedPath;

                Program.CorrectPath(ref path);

                Global.DestPath = this.destPathBox.Text = path;
            }
        }

        public void SetLog(string log)
        {
            this.Result.Text = log;
        }

        private FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
    }
}
