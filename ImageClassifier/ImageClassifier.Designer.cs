namespace ImageClassifier
{
    partial class ImageClassifier
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
            this.pbImage = new System.Windows.Forms.PictureBox();
            this.BtnAnalyzeEmotion = new System.Windows.Forms.Button();
            this.BtnSelectImageFile = new System.Windows.Forms.Button();
            this.BtnTrain = new System.Windows.Forms.Button();
            this.RtbLog = new System.Windows.Forms.RichTextBox();
            this.BtnSaveModel = new System.Windows.Forms.Button();
            this.BtnLoadModel = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pbImage)).BeginInit();
            this.SuspendLayout();
            // 
            // pbImage
            // 
            this.pbImage.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pbImage.Location = new System.Drawing.Point(0, 0);
            this.pbImage.Name = "pbImage";
            this.pbImage.Size = new System.Drawing.Size(404, 448);
            this.pbImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pbImage.TabIndex = 0;
            this.pbImage.TabStop = false;
            // 
            // BtnAnalyzeEmotion
            // 
            this.BtnAnalyzeEmotion.Location = new System.Drawing.Point(425, 162);
            this.BtnAnalyzeEmotion.Name = "BtnAnalyzeEmotion";
            this.BtnAnalyzeEmotion.Size = new System.Drawing.Size(95, 28);
            this.BtnAnalyzeEmotion.TabIndex = 1;
            this.BtnAnalyzeEmotion.Text = "Emotion Detect";
            this.BtnAnalyzeEmotion.UseVisualStyleBackColor = true;
            this.BtnAnalyzeEmotion.Click += new System.EventHandler(this.BtnAnalyzeEmotion_Click);
            // 
            // BtnSelectImageFile
            // 
            this.BtnSelectImageFile.Location = new System.Drawing.Point(425, 124);
            this.BtnSelectImageFile.Name = "BtnSelectImageFile";
            this.BtnSelectImageFile.Size = new System.Drawing.Size(95, 28);
            this.BtnSelectImageFile.TabIndex = 2;
            this.BtnSelectImageFile.Text = "Select Image";
            this.BtnSelectImageFile.UseVisualStyleBackColor = true;
            this.BtnSelectImageFile.Click += new System.EventHandler(this.BtnSelectImageFile_Click);
            // 
            // BtnTrain
            // 
            this.BtnTrain.Location = new System.Drawing.Point(425, 48);
            this.BtnTrain.Name = "BtnTrain";
            this.BtnTrain.Size = new System.Drawing.Size(95, 28);
            this.BtnTrain.TabIndex = 3;
            this.BtnTrain.Text = "Train Model";
            this.BtnTrain.UseVisualStyleBackColor = true;
            this.BtnTrain.Click += new System.EventHandler(this.BtnTrain_Click);
            // 
            // RtbLog
            // 
            this.RtbLog.Location = new System.Drawing.Point(417, 219);
            this.RtbLog.Name = "RtbLog";
            this.RtbLog.ScrollBars = System.Windows.Forms.RichTextBoxScrollBars.None;
            this.RtbLog.Size = new System.Drawing.Size(367, 218);
            this.RtbLog.TabIndex = 4;
            this.RtbLog.Text = "";
            this.RtbLog.TextChanged += new System.EventHandler(this.RtbLog_TextChanged);
            // 
            // BtnSaveModel
            // 
            this.BtnSaveModel.Location = new System.Drawing.Point(425, 85);
            this.BtnSaveModel.Name = "BtnSaveModel";
            this.BtnSaveModel.Size = new System.Drawing.Size(95, 28);
            this.BtnSaveModel.TabIndex = 5;
            this.BtnSaveModel.Text = "Save Model";
            this.BtnSaveModel.UseVisualStyleBackColor = true;
            this.BtnSaveModel.Click += new System.EventHandler(this.BtnSaveModel_Click);
            // 
            // BtnLoadModel
            // 
            this.BtnLoadModel.Location = new System.Drawing.Point(425, 12);
            this.BtnLoadModel.Name = "BtnLoadModel";
            this.BtnLoadModel.Size = new System.Drawing.Size(95, 28);
            this.BtnLoadModel.TabIndex = 6;
            this.BtnLoadModel.Text = "Load Model";
            this.BtnLoadModel.UseVisualStyleBackColor = true;
            this.BtnLoadModel.Click += new System.EventHandler(this.BtnLoadModel_Click);
            // 
            // ImageClassifier
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.BtnLoadModel);
            this.Controls.Add(this.BtnSaveModel);
            this.Controls.Add(this.RtbLog);
            this.Controls.Add(this.BtnTrain);
            this.Controls.Add(this.BtnSelectImageFile);
            this.Controls.Add(this.BtnAnalyzeEmotion);
            this.Controls.Add(this.pbImage);
            this.Name = "ImageClassifier";
            this.Text = "ImageClassifier";
            ((System.ComponentModel.ISupportInitialize)(this.pbImage)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pbImage;
        private System.Windows.Forms.Button BtnAnalyzeEmotion;
        private System.Windows.Forms.Button BtnSelectImageFile;
        private System.Windows.Forms.Button BtnTrain;
        private System.Windows.Forms.RichTextBox RtbLog;
        private System.Windows.Forms.Button BtnSaveModel;
        private System.Windows.Forms.Button BtnLoadModel;
    }
}

