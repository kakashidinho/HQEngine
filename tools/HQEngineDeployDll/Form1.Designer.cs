namespace HQEngineDeployDll
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.D3D9 = new System.Windows.Forms.CheckBox();
            this.button2 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.srcPathBox = new System.Windows.Forms.TextBox();
            this.destPathBox = new System.Windows.Forms.TextBox();
            this.D3D11 = new System.Windows.Forms.CheckBox();
            this.GL = new System.Windows.Forms.CheckBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.cg = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.Debug = new System.Windows.Forms.RadioButton();
            this.Release = new System.Windows.Forms.RadioButton();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.Math = new System.Windows.Forms.CheckBox();
            this.Util = new System.Windows.Forms.CheckBox();
            this.Engine = new System.Windows.Forms.CheckBox();
            this.Audio = new System.Windows.Forms.CheckBox();
            this.XAudio = new System.Windows.Forms.RadioButton();
            this.openAL = new System.Windows.Forms.RadioButton();
            this.Result = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(410, 339);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 0;
            this.button1.Text = "Exit";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // D3D9
            // 
            this.D3D9.AutoSize = true;
            this.D3D9.Location = new System.Drawing.Point(37, 109);
            this.D3D9.Name = "D3D9";
            this.D3D9.Size = new System.Drawing.Size(54, 17);
            this.D3D9.TabIndex = 1;
            this.D3D9.Text = "D3D9";
            this.D3D9.UseVisualStyleBackColor = true;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(1, 13);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(122, 23);
            this.button2.TabIndex = 2;
            this.button2.Text = "3rd Party Dlls Folder";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(1, 52);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(122, 23);
            this.button3.TabIndex = 3;
            this.button3.Text = "Deployment Folder";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // srcPathBox
            // 
            this.srcPathBox.Location = new System.Drawing.Point(129, 15);
            this.srcPathBox.Name = "srcPathBox";
            this.srcPathBox.ReadOnly = true;
            this.srcPathBox.Size = new System.Drawing.Size(356, 20);
            this.srcPathBox.TabIndex = 4;
            // 
            // destPathBox
            // 
            this.destPathBox.Location = new System.Drawing.Point(129, 55);
            this.destPathBox.Name = "destPathBox";
            this.destPathBox.ReadOnly = true;
            this.destPathBox.Size = new System.Drawing.Size(356, 20);
            this.destPathBox.TabIndex = 5;
            // 
            // D3D11
            // 
            this.D3D11.AutoSize = true;
            this.D3D11.Location = new System.Drawing.Point(111, 109);
            this.D3D11.Name = "D3D11";
            this.D3D11.Size = new System.Drawing.Size(60, 17);
            this.D3D11.TabIndex = 6;
            this.D3D11.Text = "D3D11";
            this.D3D11.UseVisualStyleBackColor = true;
            // 
            // GL
            // 
            this.GL.AutoSize = true;
            this.GL.Location = new System.Drawing.Point(6, 28);
            this.GL.Name = "GL";
            this.GL.Size = new System.Drawing.Size(40, 17);
            this.GL.TabIndex = 7;
            this.GL.Text = "GL";
            this.GL.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cg);
            this.groupBox1.Controls.Add(this.GL);
            this.groupBox1.Location = new System.Drawing.Point(177, 81);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(111, 61);
            this.groupBox1.TabIndex = 8;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "GL";
            // 
            // cg
            // 
            this.cg.AutoSize = true;
            this.cg.Enabled = false;
            this.cg.Location = new System.Drawing.Point(66, 28);
            this.cg.Name = "cg";
            this.cg.Size = new System.Drawing.Size(38, 17);
            this.cg.TabIndex = 8;
            this.cg.Text = "cg";
            this.cg.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.openAL);
            this.groupBox2.Controls.Add(this.XAudio);
            this.groupBox2.Controls.Add(this.Audio);
            this.groupBox2.Location = new System.Drawing.Point(306, 81);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(179, 84);
            this.groupBox2.TabIndex = 9;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Audio";
            // 
            // Debug
            // 
            this.Debug.AutoSize = true;
            this.Debug.Location = new System.Drawing.Point(28, 196);
            this.Debug.Name = "Debug";
            this.Debug.Size = new System.Drawing.Size(57, 17);
            this.Debug.TabIndex = 10;
            this.Debug.Text = "Debug";
            this.Debug.UseVisualStyleBackColor = true;
            // 
            // Release
            // 
            this.Release.AutoSize = true;
            this.Release.Checked = true;
            this.Release.Location = new System.Drawing.Point(28, 219);
            this.Release.Name = "Release";
            this.Release.Size = new System.Drawing.Size(64, 17);
            this.Release.TabIndex = 11;
            this.Release.TabStop = true;
            this.Release.Text = "Release";
            this.Release.UseVisualStyleBackColor = true;
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(18, 338);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(75, 23);
            this.button4.TabIndex = 12;
            this.button4.Text = "Remove";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(172, 337);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(125, 23);
            this.button5.TabIndex = 13;
            this.button5.Text = "Deploy";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // Math
            // 
            this.Math.AutoSize = true;
            this.Math.Location = new System.Drawing.Point(37, 158);
            this.Math.Name = "Math";
            this.Math.Size = new System.Drawing.Size(50, 17);
            this.Math.TabIndex = 14;
            this.Math.Text = "Math";
            this.Math.UseVisualStyleBackColor = true;
            // 
            // Util
            // 
            this.Util.AutoSize = true;
            this.Util.Location = new System.Drawing.Point(111, 158);
            this.Util.Name = "Util";
            this.Util.Size = new System.Drawing.Size(41, 17);
            this.Util.TabIndex = 15;
            this.Util.Text = "Util";
            this.Util.UseVisualStyleBackColor = true;
            // 
            // Engine
            // 
            this.Engine.AutoSize = true;
            this.Engine.Location = new System.Drawing.Point(183, 158);
            this.Engine.Name = "Engine";
            this.Engine.Size = new System.Drawing.Size(59, 17);
            this.Engine.TabIndex = 16;
            this.Engine.Text = "Engine";
            this.Engine.UseVisualStyleBackColor = true;
            // 
            // Audio
            // 
            this.Audio.AutoSize = true;
            this.Audio.Location = new System.Drawing.Point(7, 28);
            this.Audio.Name = "Audio";
            this.Audio.Size = new System.Drawing.Size(53, 17);
            this.Audio.TabIndex = 0;
            this.Audio.Text = "Audio";
            this.Audio.UseVisualStyleBackColor = true;
            // 
            // XAudio
            // 
            this.XAudio.AutoSize = true;
            this.XAudio.Checked = true;
            this.XAudio.Enabled = false;
            this.XAudio.Location = new System.Drawing.Point(67, 28);
            this.XAudio.Name = "XAudio";
            this.XAudio.Size = new System.Drawing.Size(59, 17);
            this.XAudio.TabIndex = 1;
            this.XAudio.TabStop = true;
            this.XAudio.Text = "XAudio";
            this.XAudio.UseVisualStyleBackColor = true;
            // 
            // openAL
            // 
            this.openAL.AutoSize = true;
            this.openAL.Enabled = false;
            this.openAL.Location = new System.Drawing.Point(67, 51);
            this.openAL.Name = "openAL";
            this.openAL.Size = new System.Drawing.Size(62, 17);
            this.openAL.TabIndex = 2;
            this.openAL.TabStop = true;
            this.openAL.Text = "openAL";
            this.openAL.UseVisualStyleBackColor = true;
            // 
            // Result
            // 
            this.Result.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.Result.Location = new System.Drawing.Point(141, 196);
            this.Result.Multiline = true;
            this.Result.Name = "Result";
            this.Result.ReadOnly = true;
            this.Result.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.Result.Size = new System.Drawing.Size(324, 109);
            this.Result.TabIndex = 17;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(497, 361);
            this.Controls.Add(this.Result);
            this.Controls.Add(this.Engine);
            this.Controls.Add(this.Util);
            this.Controls.Add(this.Math);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.Release);
            this.Controls.Add(this.Debug);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.D3D11);
            this.Controls.Add(this.destPathBox);
            this.Controls.Add(this.srcPathBox);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.D3D9);
            this.Controls.Add(this.button1);
            this.Name = "Form1";
            this.Text = "HQEngine dlls Deployment";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.CheckBox D3D9;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.TextBox srcPathBox;
        private System.Windows.Forms.TextBox destPathBox;
        private System.Windows.Forms.CheckBox D3D11;
        private System.Windows.Forms.CheckBox GL;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox cg;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.RadioButton Debug;
        private System.Windows.Forms.RadioButton Release;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.CheckBox Math;
        private System.Windows.Forms.CheckBox Util;
        private System.Windows.Forms.CheckBox Engine;
        private System.Windows.Forms.RadioButton openAL;
        private System.Windows.Forms.RadioButton XAudio;
        private System.Windows.Forms.CheckBox Audio;
        private System.Windows.Forms.TextBox Result;
    }
}

