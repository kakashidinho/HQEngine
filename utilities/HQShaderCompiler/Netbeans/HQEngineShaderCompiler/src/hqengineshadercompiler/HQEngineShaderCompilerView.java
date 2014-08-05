/*
 * HQEngineShaderCompilerView.java
 */

package hqengineshadercompiler;

import org.jdesktop.application.Action;
import org.jdesktop.application.ResourceMap;
import org.jdesktop.application.SingleFrameApplication;
import org.jdesktop.application.FrameView;
import org.jdesktop.application.TaskMonitor;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import javax.swing.Timer;
import javax.swing.Icon;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JComboBox;
import java.awt.event.* ;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * The application's main frame.
 */
public class HQEngineShaderCompilerView extends FrameView {

    private final String initFile = "setting.ini";
    private final String[] semanticsGLSL = new String[] {
        "-DVPOSITION=ATTR0",
	"-DVCOLOR=ATTR1",
	"-DVNORMAL=ATTR2",
	"-DVTEXCOORD0=ATTR3",
	"-DVTEXCOORD1=ATTR4",
	"-DVTEXCOORD2=ATTR5",
	"-DVTEXCOORD3=ATTR6",
	"-DVTEXCOORD4=ATTR7",
	"-DVTEXCOORD5=ATTR8",
	"-DVTEXCOORD6=ATTR9",
	"-DVTEXCOORD7=ATTR10",
	"-DVTANGENT=ATTR11",
	"-DVBINORMAL=ATTR12",
	"-DVBLENDWEIGHT=ATTR13",
	"-DVBLENDINDICES=ATTR14",
	"-DVPSIZE=ATTR15"
    }
    ;
    private final String[] semanticsHLSL = new String[]{
        "-DVPOSITION=POSITION",
	"-DVCOLOR=COLOR",
	"-DVNORMAL=NORMAL",
	"-DVTEXCOORD0=TEXCOORD0",
	"-DVTEXCOORD1=TEXCOORD1",
	"-DVTEXCOORD2=TEXCOORD2",
	"-DVTEXCOORD3=TEXCOORD3",
	"-DVTEXCOORD4=TEXCOORD4",
	"-DVTEXCOORD5=TEXCOORD5",
	"-DVTEXCOORD6=TEXCOORD6",
	"-DVTEXCOORD7=TEXCOORD7",
	"-DVTANGENT=TANGENT",
	"-DVBINORMAL=BINORMAL",
	"-DVBLENDWEIGHT=BLENDWEIGHT",
	"-DVBLENDINDICES=BLENDINDICES",
	"-DVPSIZE=PSIZE"
    }
    ;

    private final int SCM_CG = 0;
    private final int SCM_GLSL = 1 ;//compile source ngôn ngữ GLSL hoặc GLSL ES
    private final int SCM_HLSL = 2;//compile source directX HLSL.
    private final int SCM_CG2GLSL = 3;//translate from Cg to GLSL
    private final int SCM_CG2GLSL_ES = 4;//translate from Cg to GLSL ES
    
    
    private class TextAreaOutputStream extends OutputStream
    {
    
        @Override public void write(byte[] b ) throws IOException
        {
            outputTextArea.append(new String(b, 0, b.length));
        }
        @Override public void write(byte[] b , int offset , int length) throws IOException
        {
            outputTextArea.append(new String(b, offset, length));
        }
        @Override public void write(int b) throws IOException
        {
            outputTextArea.append(new String(new byte[]{(byte)b}));
        }
    }

    private class CMModeChangedListener implements ItemListener
    {
        @Override public void itemStateChanged(ItemEvent evt) {
            JComboBox cb = (JComboBox)evt.getSource();

            Object item = evt.getItem();

            
            if (item.equals("SCM_CG"))
                compileMode = SCM_CG;
            else if (item.equals("SCM_GLSL"))
                compileMode = SCM_GLSL;
            else if (item.equals("SCM_HLSL"))
                compileMode = SCM_HLSL;
            else if (item.equals("SCM_CG2GLSL"))
                compileMode = SCM_CG2GLSL;
            else if (item.equals("SCM_CG2GLSL_ES"))
                compileMode = SCM_CG2GLSL_ES;

            if (evt.getStateChange() == ItemEvent.SELECTED) {

                if (compileMode == SCM_CG)
                {
                    profileOptionTextField.setEnabled(true);
                    saveFileButton.setEnabled(outputCheckBox.isSelected());
                    outputCheckBox.setEnabled(true);
                    entryNameTextField.setEnabled(true);
                    profileTextField.setEnabled(true);
                    argsTextField.setEnabled(true);
                    glslVersionTextField.setEnabled(false);
                    glslShaderTypeList.setEnabled(false);
                    glslMacrosTestField.setEnabled(false);
                }
                else
                {
                    profileOptionTextField.setEnabled(false);
                    if (compileMode == SCM_GLSL)
                    {
                        saveFileButton.setEnabled(false);
                        outputCheckBox.setEnabled(false);
                        entryNameTextField.setEnabled(false);
                        entryNameTextField.setText("main");
                        profileTextField.setEnabled(false);
                        argsTextField.setEnabled(false);
                        glslVersionTextField.setEnabled(true);
                        glslShaderTypeList.setEnabled(true);
                        glslMacrosTestField.setEnabled(true);
                    }
                    else if (compileMode == SCM_HLSL)
                    {
                        saveFileButton.setEnabled(outputCheckBox.isSelected());
                        outputCheckBox.setEnabled(true);
                        entryNameTextField.setEnabled(true);
                        profileTextField.setEnabled(true);
                        argsTextField.setEnabled(true);
                        glslVersionTextField.setEnabled(false);
                        glslShaderTypeList.setEnabled(false);
                        glslMacrosTestField.setEnabled(false);
                    }
                    else if (compileMode == SCM_CG2GLSL || compileMode == SCM_CG2GLSL_ES){
                        saveFileButton.setEnabled(outputCheckBox.isSelected());
                        outputCheckBox.setEnabled(true);
                        entryNameTextField.setEnabled(true);
                        profileTextField.setEnabled(false);
                        argsTextField.setEnabled(true);
                        glslVersionTextField.setEnabled(true);
                        glslShaderTypeList.setEnabled(true);
                        glslMacrosTestField.setEnabled(false);
                    }
                }
            } else if (evt.getStateChange() == ItemEvent.DESELECTED) {
                // Item is no longer selected
            }
        }

    }
    
    public HQEngineShaderCompilerView(SingleFrameApplication app) {
        super(app);
        this.getFrame().setTitle("HQEngine Shader Compiler");
        try
        {
          javax.swing.UIManager.setLookAndFeel(javax.swing.UIManager.getSystemLookAndFeelClassName());
        }
        catch (Exception ignored) {/* ignored - will use default look and feel should this fail */}


        if (glsl_compiler_ready)
            InitGL();
        initComponents();


        this.getFrame().setResizable(false);
        this.mainPanel.setSize(this.mainPanel.getPreferredSize());

        CMModeChangedListener actionListener = new CMModeChangedListener();
        this.compileModeList.addItemListener(actionListener);


        this.Load();

        this.compileModeList.setSelectedIndex(this.compileMode);

        this.outputCheckBox.setSelected(this.outputFile);
        this.saveFileButton.setEnabled(this.outputFile);
        this.textAreaOutputStream = new TextAreaOutputStream();

        this.outputPathText.setText(this.savePath);
        this.sourcePathText.setText(this.sourcePath);

        System.setOut(new PrintStream(this.textAreaOutputStream));
        //System.setErr(new PrintStream(this.textAreaOutputStream));

        // status bar initialization - message timeout, idle icon and busy animation, etc
        ResourceMap resourceMap = getResourceMap();
        int messageTimeout = resourceMap.getInteger("StatusBar.messageTimeout");
        messageTimer = new Timer(messageTimeout, new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                statusMessageLabel.setText("");
            }
        });
        messageTimer.setRepeats(false);
        int busyAnimationRate = resourceMap.getInteger("StatusBar.busyAnimationRate");
        for (int i = 0; i < busyIcons.length; i++) {
            busyIcons[i] = resourceMap.getIcon("StatusBar.busyIcons[" + i + "]");
        }
        busyIconTimer = new Timer(busyAnimationRate, new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                busyIconIndex = (busyIconIndex + 1) % busyIcons.length;
                statusAnimationLabel.setIcon(busyIcons[busyIconIndex]);
            }
        });
        idleIcon = resourceMap.getIcon("StatusBar.idleIcon");
        statusAnimationLabel.setIcon(idleIcon);
        progressBar.setVisible(false);

        // connecting action tasks to status bar via TaskMonitor
        TaskMonitor taskMonitor = new TaskMonitor(getApplication().getContext());
        taskMonitor.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                String propertyName = evt.getPropertyName();
                if ("started".equals(propertyName)) {
                    if (!busyIconTimer.isRunning()) {
                        statusAnimationLabel.setIcon(busyIcons[0]);
                        busyIconIndex = 0;
                        busyIconTimer.start();
                    }
                    progressBar.setVisible(true);
                    progressBar.setIndeterminate(true);
                } else if ("done".equals(propertyName)) {
                    busyIconTimer.stop();
                    statusAnimationLabel.setIcon(idleIcon);
                    progressBar.setVisible(false);
                    progressBar.setValue(0);
                } else if ("message".equals(propertyName)) {
                    String text = (String)(evt.getNewValue());
                    statusMessageLabel.setText((text == null) ? "" : text);
                    messageTimer.restart();
                } else if ("progress".equals(propertyName)) {
                    int value = (Integer)(evt.getNewValue());
                    progressBar.setVisible(true);
                    progressBar.setIndeterminate(false);
                    progressBar.setValue(value);
                }
            }
        });

        this.getFrame().addWindowListener(new java.awt.event.WindowAdapter() {
               @Override   public void windowClosing(WindowEvent e)
               {
                   Save();
                   if (glsl_compiler_ready)
                        ReleaseGL();
                   System.exit(0);
               }
            }
        );
    }

    public void Load()
    {
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(this.initFile) , "US-ASCII"));

            String line;

            line = br.readLine();
            this.compileMode = Integer.parseInt(line);
            line = br.readLine();
            this.outputFile = Boolean.parseBoolean(line);

            this.sourcePath = br.readLine();
            this.savePath = br.readLine();

            this.entryNameTextField.setText(br.readLine());
            this.profileTextField.setText(br.readLine());
            this.glslVersionTextField.setText(br.readLine());
            this.glslShaderTypeList.setSelectedIndex(Integer.parseInt(br.readLine()));

            this.fileChooser.setCurrentDirectory(new File(br.readLine()));

            br.close();
        }
        catch (FileNotFoundException e)
        {
            return;
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
    public void Save()
    {

        try
        {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.initFile) , "US-ASCII"));

            bw.write(String.valueOf(this.compileMode)); bw.newLine();
            bw.write(this.outputFile? "true" : "false"); bw.newLine();

            bw.write(this.sourcePath); bw.newLine();
            bw.write(this.savePath); bw.newLine();

            bw.write(this.entryNameTextField.getText()); bw.newLine();
            bw.write(this.profileTextField.getText()); bw.newLine();
            bw.write(this.glslVersionTextField.getText()); bw.newLine();
            bw.write(String.valueOf(this.glslShaderTypeList.getSelectedIndex())); bw.newLine();

            bw.write(this.fileChooser.getCurrentDirectory().getPath());

            bw.close();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        mainPanel = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        outputTextArea = new javax.swing.JTextArea();
        jButton1 = new javax.swing.JButton();
        sourcePathText = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        outputPathText = new javax.swing.JTextField();
        jLabel3 = new javax.swing.JLabel();
        profileTextField = new javax.swing.JTextField();
        jButton2 = new javax.swing.JButton();
        outputCheckBox = new javax.swing.JCheckBox();
        saveFileButton = new javax.swing.JButton();
        profileOptionTextField = new javax.swing.JTextField();
        jLabel4 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        entryNameTextField = new javax.swing.JTextField();
        jLabel5 = new javax.swing.JLabel();
        argsTextField = new javax.swing.JTextField();
        compileButton = new javax.swing.JButton();
        compileModeList = new javax.swing.JComboBox();
        jLabel6 = new javax.swing.JLabel();
        jLabel7 = new javax.swing.JLabel();
        glslVersionTextField = new javax.swing.JTextField();
        glslShaderTypeList = new javax.swing.JComboBox();
        jLabel8 = new javax.swing.JLabel();
        jLabel9 = new javax.swing.JLabel();
        glslMacrosTestField = new javax.swing.JTextField();
        menuBar = new javax.swing.JMenuBar();
        javax.swing.JMenu fileMenu = new javax.swing.JMenu();
        javax.swing.JMenuItem exitMenuItem = new javax.swing.JMenuItem();
        statusPanel = new javax.swing.JPanel();
        javax.swing.JSeparator statusPanelSeparator = new javax.swing.JSeparator();
        statusMessageLabel = new javax.swing.JLabel();
        statusAnimationLabel = new javax.swing.JLabel();
        progressBar = new javax.swing.JProgressBar();
        fileChooser = new javax.swing.JFileChooser();

        mainPanel.setName("mainPanel"); // NOI18N
        mainPanel.setPreferredSize(new java.awt.Dimension(800, 600));

        jScrollPane1.setName("outputTextPane"); // NOI18N

        org.jdesktop.application.ResourceMap resourceMap = org.jdesktop.application.Application.getInstance(hqengineshadercompiler.HQEngineShaderCompilerApp.class).getContext().getResourceMap(HQEngineShaderCompilerView.class);
        outputTextArea.setBackground(resourceMap.getColor("outputTextArea.background")); // NOI18N
        outputTextArea.setColumns(20);
        outputTextArea.setEditable(false);
        outputTextArea.setFont(resourceMap.getFont("outputTextArea.font")); // NOI18N
        outputTextArea.setForeground(resourceMap.getColor("outputTextArea.foreground")); // NOI18N
        outputTextArea.setLineWrap(true);
        outputTextArea.setRows(5);
        outputTextArea.setName("outputTextArea"); // NOI18N
        jScrollPane1.setViewportView(outputTextArea);

        jButton1.setText(resourceMap.getString("jButton1.text")); // NOI18N
        jButton1.setActionCommand("Open");
        jButton1.setName("jButton1"); // NOI18N
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        sourcePathText.setEditable(false);
        sourcePathText.setText(resourceMap.getString("sourcePathText.text")); // NOI18N
        sourcePathText.setName("sourcePathText"); // NOI18N

        jLabel1.setText(resourceMap.getString("jLabel1.text")); // NOI18N
        jLabel1.setName("jLabel1"); // NOI18N

        outputPathText.setEditable(false);
        outputPathText.setName("outputPathText"); // NOI18N

        jLabel3.setText(resourceMap.getString("jLabel3.text")); // NOI18N
        jLabel3.setName("jLabel3"); // NOI18N

        profileTextField.setText(resourceMap.getString("profileTextField.text")); // NOI18N
        profileTextField.setName("profileTextField"); // NOI18N

        jButton2.setText(resourceMap.getString("jButton2.text")); // NOI18N
        jButton2.setName("jButton2"); // NOI18N
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        outputCheckBox.setText(resourceMap.getString("outputCheckBox.text")); // NOI18N
        outputCheckBox.setName("outputCheckBox"); // NOI18N
        outputCheckBox.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                outputCheckBoxStateChanged(evt);
            }
        });

        saveFileButton.setText(resourceMap.getString("saveFileButton.text")); // NOI18N
        saveFileButton.setName("saveFileButton"); // NOI18N
        saveFileButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveFileButtonActionPerformed(evt);
            }
        });

        profileOptionTextField.setText(resourceMap.getString("profileOptionTextField.text")); // NOI18N
        profileOptionTextField.setName("profileOptionTextField"); // NOI18N

        jLabel4.setText(resourceMap.getString("jLabel4.text")); // NOI18N
        jLabel4.setName("jLabel4"); // NOI18N

        jLabel2.setText(resourceMap.getString("jLabel2.text")); // NOI18N
        jLabel2.setName("jLabel2"); // NOI18N

        entryNameTextField.setText(resourceMap.getString("entryNameTextField.text")); // NOI18N
        entryNameTextField.setName("entryNameTextField"); // NOI18N

        jLabel5.setText(resourceMap.getString("jLabel5.text")); // NOI18N
        jLabel5.setName("jLabel5"); // NOI18N

        argsTextField.setText(resourceMap.getString("argsTextField.text")); // NOI18N
        argsTextField.setName("argsTextField"); // NOI18N

        compileButton.setText(resourceMap.getString("compileButton.text")); // NOI18N
        compileButton.setToolTipText(resourceMap.getString("compileButton.toolTipText")); // NOI18N
        compileButton.setName("compileButton"); // NOI18N
        compileButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                compileButtonActionPerformed(evt);
            }
        });

        compileModeList.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "SCM_CG", "SCM_GLSL", "SCM_HLSL", "SCM_CG2GLSL", "SCM_CG2GLSL_ES" }));
        compileModeList.setName("compileModeList"); // NOI18N

        jLabel6.setText(resourceMap.getString("jLabel6.text")); // NOI18N
        jLabel6.setName("jLabel6"); // NOI18N

        jLabel7.setText(resourceMap.getString("jLabel7.text")); // NOI18N
        jLabel7.setName("jLabel7"); // NOI18N

        glslVersionTextField.setText(resourceMap.getString("glslVersionTextField.text")); // NOI18N
        glslVersionTextField.setEnabled(false);
        glslVersionTextField.setName("glslVersionTextField"); // NOI18N

        glslShaderTypeList.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "GL_VERTEX_SHADER", "GL_GEOMETRY_SHADER", "GL_FRAGMENT_SHADER", "GL_TESS_CONTROL_SHADER", "GL_TESS_EVALUATION_SHADER", "GL_COMPUTE_SHADER" }));
        glslShaderTypeList.setEnabled(false);
        glslShaderTypeList.setName("glslShaderTypeList"); // NOI18N

        jLabel8.setText(resourceMap.getString("jLabel8.text")); // NOI18N
        jLabel8.setName("jLabel8"); // NOI18N

        jLabel9.setText(resourceMap.getString("jLabel9.text")); // NOI18N
        jLabel9.setName("jLabel9"); // NOI18N

        glslMacrosTestField.setText(resourceMap.getString("glslMacrosTestField.text")); // NOI18N
        glslMacrosTestField.setEnabled(false);
        glslMacrosTestField.setName("glslMacrosTestField"); // NOI18N

        javax.swing.GroupLayout mainPanelLayout = new javax.swing.GroupLayout(mainPanel);
        mainPanel.setLayout(mainPanelLayout);
        mainPanelLayout.setHorizontalGroup(
            mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(mainPanelLayout.createSequentialGroup()
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(mainPanelLayout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 1028, Short.MAX_VALUE))
                    .addGroup(mainPanelLayout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(outputCheckBox, javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(jLabel1, javax.swing.GroupLayout.Alignment.TRAILING))
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, mainPanelLayout.createSequentialGroup()
                                .addGap(18, 18, 18)
                                .addComponent(sourcePathText, javax.swing.GroupLayout.PREFERRED_SIZE, 392, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(mainPanelLayout.createSequentialGroup()
                                .addGap(18, 18, 18)
                                .addComponent(outputPathText, javax.swing.GroupLayout.PREFERRED_SIZE, 392, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(jButton1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(saveFileButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addGap(305, 305, 305))
                    .addGroup(mainPanelLayout.createSequentialGroup()
                        .addGap(416, 416, 416)
                        .addComponent(jButton2))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, mainPanelLayout.createSequentialGroup()
                        .addGap(28, 28, 28)
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel3)
                            .addComponent(jLabel4)
                            .addComponent(jLabel2))
                        .addGap(24, 24, 24)
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(profileOptionTextField, javax.swing.GroupLayout.DEFAULT_SIZE, 283, Short.MAX_VALUE)
                            .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addComponent(entryNameTextField, javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(profileTextField, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, 93, Short.MAX_VALUE)))
                        .addGap(18, 18, 18)
                        .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(compileButton)
                            .addGroup(mainPanelLayout.createSequentialGroup()
                                .addComponent(jLabel5)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(argsTextField, javax.swing.GroupLayout.DEFAULT_SIZE, 454, Short.MAX_VALUE))
                            .addGroup(mainPanelLayout.createSequentialGroup()
                                .addComponent(jLabel9)
                                .addGap(18, 18, 18)
                                .addComponent(glslMacrosTestField))
                            .addGroup(mainPanelLayout.createSequentialGroup()
                                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                    .addGroup(mainPanelLayout.createSequentialGroup()
                                        .addComponent(jLabel7)
                                        .addGap(18, 18, 18)
                                        .addComponent(glslVersionTextField))
                                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, mainPanelLayout.createSequentialGroup()
                                        .addComponent(jLabel6)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(compileModeList, javax.swing.GroupLayout.PREFERRED_SIZE, 176, javax.swing.GroupLayout.PREFERRED_SIZE)))
                                .addGap(54, 54, 54)
                                .addComponent(jLabel8)
                                .addGap(18, 18, 18)
                                .addComponent(glslShaderTypeList, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))))
                .addContainerGap())
        );
        mainPanelLayout.setVerticalGroup(
            mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(mainPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 206, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(5, 5, 5)
                .addComponent(jButton2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(sourcePathText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1)
                    .addComponent(jButton1))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(outputPathText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(outputCheckBox)
                    .addComponent(saveFileButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 106, Short.MAX_VALUE)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel6)
                    .addComponent(compileModeList, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel2)
                    .addComponent(entryNameTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel7)
                    .addComponent(glslShaderTypeList, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel8)
                    .addComponent(glslVersionTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel3)
                    .addComponent(profileTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel9)
                    .addComponent(glslMacrosTestField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(mainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel4)
                    .addComponent(profileOptionTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel5)
                    .addComponent(argsTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addComponent(compileButton)
                .addGap(6, 6, 6))
        );

        menuBar.setName("menuBar"); // NOI18N

        fileMenu.setText(resourceMap.getString("fileMenu.text")); // NOI18N
        fileMenu.setName("fileMenu"); // NOI18N

        javax.swing.ActionMap actionMap = org.jdesktop.application.Application.getInstance(hqengineshadercompiler.HQEngineShaderCompilerApp.class).getContext().getActionMap(HQEngineShaderCompilerView.class, this);
        exitMenuItem.setAction(actionMap.get("quit")); // NOI18N
        exitMenuItem.setName("exitMenuItem"); // NOI18N
        exitMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                exitMenuItemActionPerformed(evt);
            }
        });
        fileMenu.add(exitMenuItem);

        menuBar.add(fileMenu);

        statusPanel.setName("statusPanel"); // NOI18N

        statusPanelSeparator.setName("statusPanelSeparator"); // NOI18N

        statusMessageLabel.setName("statusMessageLabel"); // NOI18N

        statusAnimationLabel.setHorizontalAlignment(javax.swing.SwingConstants.LEFT);
        statusAnimationLabel.setName("statusAnimationLabel"); // NOI18N

        progressBar.setName("progressBar"); // NOI18N

        javax.swing.GroupLayout statusPanelLayout = new javax.swing.GroupLayout(statusPanel);
        statusPanel.setLayout(statusPanelLayout);
        statusPanelLayout.setHorizontalGroup(
            statusPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(statusPanelSeparator, javax.swing.GroupLayout.DEFAULT_SIZE, 1048, Short.MAX_VALUE)
            .addGroup(statusPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(statusMessageLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 878, Short.MAX_VALUE)
                .addComponent(progressBar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(statusAnimationLabel)
                .addContainerGap())
        );
        statusPanelLayout.setVerticalGroup(
            statusPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(statusPanelLayout.createSequentialGroup()
                .addComponent(statusPanelSeparator, javax.swing.GroupLayout.PREFERRED_SIZE, 2, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(statusPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(statusMessageLabel)
                    .addComponent(statusAnimationLabel)
                    .addComponent(progressBar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(3, 3, 3))
        );

        fileChooser.setName("fileChooser"); // NOI18N

        setComponent(mainPanel);
        setMenuBar(menuBar);
        setStatusBar(statusPanel);
    }// </editor-fold>//GEN-END:initComponents

    private void exitMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_exitMenuItemActionPerformed
        // TODO add your handling code here:
        this.Save();
        this.ReleaseGL();
        System.exit(0);
}//GEN-LAST:event_exitMenuItemActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
        int returnVal = this.fileChooser.showOpenDialog(this.getFrame());
        if (returnVal == JFileChooser.APPROVE_OPTION)
        {
            File file = this.fileChooser.getSelectedFile();
            this.sourcePath = file.getPath();
            this.sourcePathText.setText(sourcePath);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        this.outputTextArea.setText("");
    }//GEN-LAST:event_jButton2ActionPerformed

    private void outputCheckBoxStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_outputCheckBoxStateChanged
        // TODO add your handling code here:
        this.outputFile = this.outputCheckBox.isSelected();
        if (this.compileMode != SCM_GLSL)
            this.saveFileButton.setEnabled(this.outputFile);
    }//GEN-LAST:event_outputCheckBoxStateChanged

    private void saveFileButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveFileButtonActionPerformed
        // TODO add your handling code here:
        int returnVal = this.fileChooser.showSaveDialog(this.getFrame());
        if (returnVal == JFileChooser.APPROVE_OPTION)
        {
            File file = this.fileChooser.getSelectedFile();
            this.savePath = file.getPath();
            this.outputPathText.setText(savePath);
        }
    }//GEN-LAST:event_saveFileButtonActionPerformed

    private void compileButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_compileButtonActionPerformed
        // TODO add your handling code here:
        if (this.sourcePath.equals(""))
        {
            javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "source file name string is empty!",
                    "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
            return ;
        }
        
        if (this.outputFile)
        {
            if(this.savePath.equals(""))
            {
                javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "save file name string is empty!",
                        "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                return ;
            }
            else {
                //check if file exists
                File outFileHandle = new File(this.savePath);
                if (outFileHandle.exists())
                {
                    if(javax.swing.JOptionPane.showConfirmDialog(this.getFrame(), "destination file exits. overwrite?") != javax.swing.JOptionPane.OK_OPTION)
                        return;
                }
            }
        }
        
        
        Process p = null;
        StringTokenizer st = new StringTokenizer(this.argsTextField.getText());
        ArrayList<String> args = new ArrayList<String>();
        args.add("cgc");
        while(st.hasMoreTokens())
            args.add(st.nextToken());
        String profile = "";
        boolean d3d11profile = false;
        boolean d3d9profile = false;
        String glslMacros = "";
        String glslResult;
        String[] cmd;
        ProcessBuilder pb;
        try {
            switch (compileMode)
            {
                case SCM_CG:
                    if (this.entryNameTextField.getText().equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "entry name string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    profile = this.profileTextField.getText();
                    if (profile.equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "profile string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    else if (profile.equals("glslv"))
                    {
                        for (int i = 0 ; i < semanticsGLSL.length ; ++i)
                            args.add(semanticsGLSL[i]);
                    }
                    else
                    {
                        for (int i = 0 ; i < semanticsHLSL.length ; ++i)
                            args.add(semanticsHLSL[i]);
                        
                        //find shader model
                        Scanner scanner = new Scanner(profile);
                        scanner.useDelimiter("_");
                        
                        String stage = scanner.next();
                        if (stage.equals("vs") || stage.equals("ps")
                                || stage.equals("gs") || stage.equals("hs")
                                || stage.equals("ds"))
                        {
                            int major = -1, minor;
                            try {
                                major = scanner.nextInt();
                                minor = scanner.nextInt();
                            }
                            catch (Exception e)
                            {
                                major = -1;
                            }
                            
                            if (major > 3)
                                d3d11profile = true;
                            else if (major > 0)
                                d3d9profile = true;
                        }
                    }
                    args.add("-profile");
                    args.add(profile);
                    args.add("-entry");
                    args.add(this.entryNameTextField.getText());
                    args.add("-DHQEXT_CG");
                    if (d3d11profile)
                        args.add("-DHQEXT_CG_D3D11");
                    else if (d3d9profile)
                        args.add("-DHQEXT_CG_D3D9");
                        

                    st = new StringTokenizer(this.profileOptionTextField.getText());
                    if (st.hasMoreTokens())
                    {
                        args.add("-po");
                        StringBuilder sb = new StringBuilder();
                        do
                        {
                            sb.append(st.nextToken());
                            sb.append(",");
                        }while (st.hasMoreTokens());
                        args.add(sb.toString());
                    }
                    

                    if (this.outputFile)
                    {
                        args.add("-o");
                        if (isWindows)
                            args.add("\"" + this.savePath + "\"");
                        else
                            args.add(this.savePath.replaceAll(" ", "\\ "));
                    }
                    if (isWindows)
                           args.add("\"" + this.sourcePath + "\"");
                    else
                        args.add(this.sourcePath.replaceAll(" ", "\\ "));
                    cmd = new String[args.size()];
                    args.toArray(cmd);
                    pb = new ProcessBuilder(cmd);
                    pb.redirectErrorStream(true);
                    
                    
                    p = pb.start();
                    break;
                case SCM_GLSL:
                    if (!glsl_compiler_ready){
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "could not load glsl compiler!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    /*
                    if (this.glslVersionTextField.getText().equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "glsl version string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }*/
                    glslMacros = this.glslMacrosTestField.getText();
                    glslMacros = glslMacros.replace(',', '\n');
                    glslMacros += "\n";

                    glslResult = this.CompileGLSL(this.sourcePath, this.glslVersionTextField.getText(),
                            glslMacros, this.glslShaderTypeList.getSelectedIndex());

                    this.outputTextArea.setText("");
                    
                    System.out.println(glslResult);


                    break;
                case SCM_HLSL:
                    if (!isWindows)
                    {
                        System.out.println("this mode is only supported in Microsoft Windows OS!");
                        break;
                    }
                    args.set(0, "fxc");
                    if (this.entryNameTextField.getText().equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "entry name string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    profile = this.profileTextField.getText();
                    if (profile.equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "profile string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    args.add("/T");
                    args.add(profile);
                    args.add("/E");
                    args.add(this.entryNameTextField.getText());


                    if (this.outputFile)
                    {
                        args.add("/Fo");
                        args.add("\"" + this.savePath + "\"");
                    }

                    args.add("\"" + this.sourcePath + "\"");
                    cmd = new String[args.size()];
                    args.toArray(cmd);
                    pb = new ProcessBuilder(cmd);
                    pb.redirectErrorStream(true);

                    p = pb.start();
                    break;
                case SCM_CG2GLSL: case SCM_CG2GLSL_ES:
                {
                    if (this.entryNameTextField.getText().equals(""))
                    {
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "entry name string is empty!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    String shaderType = (String)this.glslShaderTypeList.getSelectedItem();
                    profile = "";
                    
                    
                    if (shaderType.equals("GL_VERTEX_SHADER"))
                    {
                        profile = "glslv";
                    }
                    else if (shaderType.equals("GL_FRAGMENT_SHADER"))
                    {
                        profile = "glslf";
                    }
                    else{
                        javax.swing.JOptionPane.showMessageDialog(this.getFrame() , "unsupported shader type in this mode!",
                            "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
                        break ;
                    }
                    String glslVersion = this.glslVersionTextField.getText();
                    if (profile.length() > 0)
                    {
                        args.add("-profile");
                        args.add(profile);
                    }
                    if (glslVersion.length() > 0)
                    {
                        args.add("-version");
                        args.add(glslVersion);
                    }
                    if (compileMode == SCM_CG2GLSL_ES)
                        args.add("-glsles");
                    
                    args.add("-entry");
                    args.add(this.entryNameTextField.getText());
                    

                    if (this.outputFile)
                    {
                        args.add("-o");
                        if (isWindows)
                            args.add("\"" + this.savePath + "\"");
                        else
                            args.add(this.savePath.replaceAll(" ", "\\ "));
                    }
                    if (isWindows)
                           args.add("\"" + this.sourcePath + "\"");
                    else
                        args.add(this.sourcePath.replaceAll(" ", "\\ "));
                    
                    args.set(0, "HQEXT_cg2glsl");
                    cmd = new String[args.size()];
                    args.toArray(cmd);
                    pb = new ProcessBuilder(cmd);
                    pb.redirectErrorStream(true);
                    
                    
                    p = pb.start();
                }
                    break;
            }
            if (p != null)
            {
                StringBuffer string = new StringBuffer();
                int c;
                BufferedReader in;
                /*
                in = new BufferedReader(
                                new InputStreamReader(p.getErrorStream()) );
                while ((c = in.read()) != -1) {
                    string.append((char)c);
                }
                in.close();

                string.append("\r\n");
                */
                in = new BufferedReader(
                                new InputStreamReader(p.getInputStream()) );
                while ((c = in.read()) != -1) {
                    string.append((char)c);
                }
                in.close();

                this.outputTextArea.setText("");
                System.out.println(string);

            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        
    }//GEN-LAST:event_compileButtonActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField argsTextField;
    private javax.swing.JButton compileButton;
    private javax.swing.JComboBox compileModeList;
    private javax.swing.JTextField entryNameTextField;
    private javax.swing.JFileChooser fileChooser;
    private javax.swing.JTextField glslMacrosTestField;
    private javax.swing.JComboBox glslShaderTypeList;
    private javax.swing.JTextField glslVersionTextField;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JPanel mainPanel;
    private javax.swing.JMenuBar menuBar;
    private javax.swing.JCheckBox outputCheckBox;
    private javax.swing.JTextField outputPathText;
    private javax.swing.JTextArea outputTextArea;
    private javax.swing.JTextField profileOptionTextField;
    private javax.swing.JTextField profileTextField;
    private javax.swing.JProgressBar progressBar;
    private javax.swing.JButton saveFileButton;
    private javax.swing.JTextField sourcePathText;
    private javax.swing.JLabel statusAnimationLabel;
    private javax.swing.JLabel statusMessageLabel;
    private javax.swing.JPanel statusPanel;
    // End of variables declaration//GEN-END:variables

    private int compileMode = SCM_CG;
    private String sourcePath = "";
    private String savePath = "";
    private boolean outputFile = false;
    private final TextAreaOutputStream textAreaOutputStream;
    private final Timer messageTimer;
    private final Timer busyIconTimer;
    private final Icon idleIcon;
    private final Icon[] busyIcons = new Icon[15];
    private int busyIconIndex = 0;

    private static boolean isWindows;
    private static boolean glsl_compiler_ready = false;
    
    private native boolean InitGL();
    private native void ReleaseGL();
    private native String CompileGLSL(String fileName , String version , String macros , int shaderType );

    static
    {
        String osName = System.getProperty("os.name");
        isWindows = (osName.indexOf("Windows") != -1 || osName.indexOf("windows") != -1);
        
        String cwd = System.getProperty("user.dir");
        char lastChar;
        if ((lastChar = cwd.charAt(cwd.length() - 1)) != '/')
            cwd += "/";

        try{
            try {
                System.load(cwd + "HQEXT_glsl_compiler.dll");
            }catch (java.lang.UnsatisfiedLinkError ee){
                //try to load x64 version
                System.out.println("loading x64 version of glsl compiler");
                System.load(cwd + "HQEXT_glsl_compiler_x64.dll");
            }
            
            glsl_compiler_ready = true;
        }
        catch(java.lang.UnsatisfiedLinkError e)
        {
            e.printStackTrace();
            
            try
            {
                System.load(cwd + "libHQEXT_glsl_compiler.jnilib");
                
                glsl_compiler_ready = true;
            }
            catch(java.lang.UnsatisfiedLinkError e2)
            {
                e2.printStackTrace();
                
                try{
                    System.load("/usr/local/lib/libHQEXT_glsl_compiler.jnilib");
                    
                    glsl_compiler_ready = true;
                }
                catch (java.lang.UnsatisfiedLinkError e3){
                    e3.printStackTrace();
                }
            }
        }
    }

}
