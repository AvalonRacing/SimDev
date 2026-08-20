package Scripts;

// TODO remove x-Axis limits for plots so that every plot is plotted completely
// TODO remove all unused code

import star.base.neo.ClientServerObject;
import star.base.neo.DoubleVector;
import star.base.neo.IntVector;
import star.base.neo.NeoObjectVector;
import star.base.query.*;
import star.base.report.ReportMonitor;
import star.cadmodeler.ExportedCartesianCoordinateSystem;
import star.common.*;
import star.flow.MassFlowReport;
import star.motion.ReferenceFrameManager;
import star.motion.UserRotatingReferenceFrame;
import star.vis.*;

import java.io.*;
import java.util.*;

public class Evalquery extends StarMacro {

    private static String postProDirStr;

    // used paramters to parse coordinate system names from simConfig.txt file
    private final String SIMCONFIG_FILE_NAME = "simConfig.txt";
    private final String KEY_VALUE_DELIMITER = "=";
    private final String PARAMETER_DELIMITER = ";";
    private final String VALUE_DELIMITER = ",";
    private final String DETAILED_PLANE_EXPORTS_KEY = "detailed_planes";
    private final String COLORRMAP_KEY="eval_colormap";

    // declaration of fixed names used in simulation. Do this globally here to see used names right away and dont have to search for them in code.
    private static final String CARNAME = "TC10";
    private static final String MRFNAME = "MRF";

    // fixed filenames used in this script
    //private static final String FILE_MONITORNAMES1_NAME = "MonitorNamesCAR1.txt";
    private static final String FILE_MONITORNAMES_NAME = "MonitorNamesCAR.txt";


    // mass flow report names wheels
    //private static final String MFR_RAD_LEFT_REPORT_NAME = "RAD_left_MF";
    //private static final String MFR_RAD_RIGHT_REPORT_NAME = "RAD_right_MF";


    // names of all used regions
    private static final String REGION_FLUID_NAME = "Air";

    // names of all used boundaries
    private static final String BOUNDARY_INLET_NAME = "Air.Inlet";

    // field functions
    private static final String FF_PRESSURECOEFFICIENT_NAME = "PressureCoefficient";
    private static String FF_CPT_NAME = "cpT";
    private static String FF_VORT_X_NAME;
    private static String FF_VORT_Y_NAME;
    private static String FF_VORT_Z_NAME;
    private static String FF_PRESSUREGRADIENT_NAME;
    private static final String FF_WALLSHEARSTRESS_NAME = "WallShearStress";

    private static final String U_INLET_PARAM_NAME = "1_velocity";

    // other constants

    // defines if scenes should be kept open after exporting planes
    private boolean keepVis = false;

    // z height of viewpoint and focalpoint used for creation of plane export
    private static final double xyPlotHeight = 0.1;
    private static final String ROTATE_COORDSYS_NAME = "PostPro";
    private static ExportedCartesianCoordinateSystem rotateCoordSystem;

    // cornering related stuff
    private final String SIMSTATE_PARAM_NAME = "90_simstate";
    private final String ROTATING_AIR_RF_NAME = "Domain";
    private boolean isCornering;

    private Simulation activeSim;
    private HashSet regions;

    public void execute() {

        activeSim = getActiveSimulation();

        String sessionPath = activeSim.getSessionPath();
        postProDirStr = sessionPath.replace(sessionPath.substring(sessionPath.lastIndexOf(File.separator)), "") + File.separator + "PostPro";

        // see if case is in cornering mode
        isCornering = ((ScalarGlobalParameter) activeSim.get(GlobalParameterManager.class).getObject(SIMSTATE_PARAM_NAME)).getQuantity().getRawValue() == 2.0;

        activeSim.println("Active SessionDir : " + activeSim.getSessionDir());
        activeSim.println("Exporting eval results to " + postProDirStr);

        // create postProDir and make directory
        File postProDir = new File(postProDirStr);
        if (postProDir.exists()) {
            activeSim.println("/PostPro directory already exists.");
        } else if (!postProDir.mkdirs()) {
            activeSim.println("Could not create /PostPro directory. Aborting eval script");
            return;
        }

        Query carquery = new Query(new CompoundPredicate(CompoundOperator.And, Arrays.asList(new NamePredicate(NameOperator.DoesNotContain, MRFNAME), new NamePredicate(NameOperator.Contains, CARNAME))), Query.STANDARD_MODIFIERS);
       
        // Double Monitors
        ArrayList<String> monitorNames = readFile(activeSim, activeSim.getSessionDir(), FILE_MONITORNAMES_NAME);


        // getting viewing directions of plane cuts based on direction of axes of para_rotate coord system
        DoubleVector focalPointXPlanes;
        DoubleVector focalPointYPlanes;
        DoubleVector focalPointZPlanes;

        DoubleVector viewPointXPlanes;
        DoubleVector viewPointYPlanes;
        DoubleVector viewPointZPlanes;

        // reading the Colormap specified in the simConfig, if nothing is specified "thermal" will be used
        String colormap = readColormap();


        try {

            LabCoordinateSystem labCoordinateSystem = activeSim.getCoordinateSystemManager().getLabCoordinateSystem();

            rotateCoordSystem = ((ExportedCartesianCoordinateSystem) labCoordinateSystem.getLocalCoordinateSystemManager().getObject(ROTATE_COORDSYS_NAME));

            double focalYStartPosX = 0.0;

            focalPointXPlanes = new DoubleVector(new double[]{0, -0.0, xyPlotHeight});
            focalPointYPlanes = new DoubleVector(new double[]{focalYStartPosX, -0.0, xyPlotHeight});
            focalPointZPlanes = new DoubleVector(new double[]{0.0, 0, 0});

            viewPointXPlanes = new DoubleVector(new double[]{-1, 0, xyPlotHeight});
            viewPointYPlanes = new DoubleVector(new double[]{focalYStartPosX, -1, xyPlotHeight});
            viewPointZPlanes = new DoubleVector(new double[]{0.0, 0, 1});

            //converting view and Focal Points with rotate Origin:
            viewPointXPlanes = rotateOrigin(viewPointXPlanes);
            viewPointYPlanes = rotateOrigin(viewPointYPlanes);
            viewPointZPlanes = rotateOrigin(viewPointZPlanes);

            focalPointXPlanes = rotateOrigin(focalPointXPlanes);
            focalPointYPlanes = rotateOrigin(focalPointYPlanes);
            focalPointZPlanes = rotateOrigin(focalPointZPlanes);
            activeSim.println("Converted focal Points and viewPoints to the correct ones w.r.t. the vehicle coord. system through rotation.");


        } catch (Exception e) {
            activeSim.println("!!! Error getting " + ROTATE_COORDSYS_NAME + " coordinate system. Aborting eval script!\nError:\n" + e);
            return;
        }

        // Einlesen der Regions in eine Collection bzw. danach zum region hashset
        Query regionquery = new Query(new TypePredicate(TypeOperator.Is, Region.class), Query.STANDARD_MODIFIERS);
        QueryResult queryResult = regionquery.createResults(activeSim);
        Collection<ClientServerObject> regionsColl = queryResult.getObjects();
        regions = new HashSet();
        regions.addAll(regionsColl);


        //--------------------------------------------------------------------------
        //Ausfuehrungen
        // Auslesen des Versionsnamens aus dem Simulationsnamen
        String versionsName = activeSim.getPresentationName();
        int versionsName_length = versionsName.length();
        versionsName = versionsName.substring(5, versionsName_length);
        // TODO insert iterations_to_average_parameter from simCOnfig.txt here
        averageReports(versionsName, 100, monitorNames, activeSim);

        activeSim.println("Starting plot exports----------------------------------------------------------");
        //coupled solver plots:
        //exportSpecPlot(activeSim, keepVis, "CFL Monitor Plot",new double[] {0, 300},null);
        //exportSpecPlot(activeSim, keepVis, "Max_ED_ERF Monitor Plot",new double[] {0.2,2},null);
        String plotDir = postProDirStr + File.separator + "Plots";
        File plotDirFile = new File(plotDir);
        if (plotDirFile.exists() && Objects.requireNonNull(plotDirFile.listFiles()).length >= 10) {
            activeSim.println("Plot directory already exists and contains files. Skipping. If you want to export plots again, delete directory and run again");
        } else {
            //Independent plots
            exportSpecPlot(activeSim, keepVis, "cD Monitor Plot", plotDir, null, null);
            exportSpecPlot(activeSim, keepVis, "cL Monitor Plot", plotDir, null, null);
            exportSpecPlot(activeSim, keepVis, "Drag Monitor Plot", plotDir, new double[]{0, 10}, null);
            exportSpecPlot(activeSim, keepVis, "Lift Monitor Plot", plotDir, new double[]{0, -10}, null);
            exportSpecPlot(activeSim, keepVis, "Residuals", plotDir, new double[]{5.0E-9, 1000}, null);
            //eportSpecPlot(activeSim, keepVis, "Solver_Iteration_Time Monitor Plot", plotDir, new double[]{0, 50}, null);
            //exportSpecPlot(activeSim, keepVis, "Solver_Total_Time Monitor Plot", plotDir, new double[]{1, 10000}, null);
            activeSim.println("Done Plotting-------------------------------------------------------------------");
        }

        // Plane Export

        // Create vorticity field functions, if not present already
	    boolean VorticityAbsolute_X_Exists = activeSim.getFieldFunctionManager().hasFunction("VorticityAbsolute_X");
    	boolean VorticityAbsolute_Y_Exists = activeSim.getFieldFunctionManager().hasFunction("VorticityAbsolute_Y");
	    boolean VorticityAbsolute_Z_Exists = activeSim.getFieldFunctionManager().hasFunction("VorticityAbsolute_Z");

        boolean PressureGradient_Exists = activeSim.getFieldFunctionManager().hasFunction("PressureGradient");


    	if (VorticityAbsolute_X_Exists == false) {

            UserFieldFunction VorticityAbsolute_X = activeSim.getFieldFunctionManager().createFieldFunction();
            VorticityAbsolute_X.getTypeOption().setSelected(FieldFunctionTypeOption.Type.SCALAR);
            VorticityAbsolute_X.setDefinition("abs($${VorticityVector}[0])");
            VorticityAbsolute_X.setFunctionName("VorticityAbsolute_X");
            VorticityAbsolute_X.setPresentationName("Vorticity Absolute X Value");
            FF_VORT_X_NAME = VorticityAbsolute_X.getFunctionName();
	    }
	    if (VorticityAbsolute_Y_Exists == false) {
	        UserFieldFunction VorticityAbsolute_Y = activeSim.getFieldFunctionManager().createFieldFunction();
            VorticityAbsolute_Y.getTypeOption().setSelected(FieldFunctionTypeOption.Type.SCALAR);
            VorticityAbsolute_Y.setDefinition("abs($${VorticityVector}[1])");
            VorticityAbsolute_Y.setFunctionName("VorticityAbsolute_Y");
            VorticityAbsolute_Y.setPresentationName("Vorticity Absolute Y Value");
            FF_VORT_Y_NAME = VorticityAbsolute_Y.getFunctionName();
	    }
	    if (VorticityAbsolute_Z_Exists == false) {
	        UserFieldFunction VorticityAbsolute_Z = activeSim.getFieldFunctionManager().createFieldFunction();
            VorticityAbsolute_Z.getTypeOption().setSelected(FieldFunctionTypeOption.Type.SCALAR);
            VorticityAbsolute_Z.setDefinition("abs($${VorticityVector}[2])");
            VorticityAbsolute_Z.setFunctionName("VorticityAbsolute_Z");
            VorticityAbsolute_Z.setPresentationName("Vorticity Absolute Z Value");
            FF_VORT_Z_NAME = VorticityAbsolute_Z.getFunctionName();
	    }

        // Create pressuregradient field functions, if not present already

        if (PressureGradient_Exists == false) {
	        UserFieldFunction PressureGradient = activeSim.getFieldFunctionManager().createFieldFunction();
            PressureGradient.getTypeOption().setSelected(FieldFunctionTypeOption.Type.VECTOR);
            PressureGradient.setDefinition("(grad(${PressureCoefficient})*[1,0,0]*$${WallShearStress}+grad(${PressureCoefficient})*[0,1,0]*$${WallShearStress}+grad(${PressureCoefficient})*[0,0,1]*$${WallShearStress})/mag($${WallShearStress}) ");
            PressureGradient.setFunctionName("PressureGradient");
            PressureGradient.setPresentationName("Pressuregradient Surface Vektor");
            FF_PRESSUREGRADIENT_NAME = PressureGradient.getFunctionName();
	    }
        
        /*
	    SceneExportParameters vortXScenesParams = new SceneExportParameters(
                FF_VORT_X_NAME,
                "vort_X",
                0,
                2000,
                0.01,
                -0.3,
                0.4,
                focalPointXPlanes,
                viewPointXPlanes,
                0.1
        );

        planeExportWParams(activeSim, vortXScenesParams, regions, true, false, colormap);


	    SceneExportParameters vortYScenesParams = new SceneExportParameters(
                FF_VORT_Y_NAME,
                "vort_Y",
                0,
                2000,
                0.01,
                -0.15,
                0.15,
                focalPointYPlanes,
                viewPointYPlanes,
                0.2
        );

        planeExportWParams(activeSim, vortYScenesParams, regions, true, false, colormap);

	    SceneExportParameters vortZScenesParams = new SceneExportParameters(
                FF_VORT_Z_NAME,
                "vort_Z",
                0,
                2000,
                0.01,
                0,
                0.15,
                focalPointZPlanes,
                viewPointZPlanes,
                0.2
        );

        planeExportWParams(activeSim, vortZScenesParams, regions, true, false, colormap);
        */
        SceneExportParameters cpXScenesParams = new SceneExportParameters(
                "PressureCoefficient",
                "cp_X",
                -0.75,
                1.0,
                0.01,
                -0.3,
                0.4,
                focalPointXPlanes,
                viewPointXPlanes,
                0.1
        );

        planeExportWParams(activeSim, cpXScenesParams, regions, true, false, colormap);

        SceneExportParameters cpYSceneParams = new SceneExportParameters(
                "PressureCoefficient",
                "cp_Y",
                -0.75,
                1.0,
                0.01,
                -0.15,
                0.15,
                focalPointYPlanes,
                viewPointYPlanes,
                0.2
        );

        planeExportWParams(activeSim, cpYSceneParams, regions, true, false, colormap);
        
        SceneExportParameters cpZSceneParams = new SceneExportParameters(
                "PressureCoefficient",
                "cp_Z",
                -0.75,
                1.0,
                0.01,
                0,
                0.15,
                focalPointZPlanes,
                viewPointZPlanes,
                0.2
        );
        planeExportWParams(activeSim, cpZSceneParams, regions, true, false, colormap);
        
        SceneExportParameters cpTXSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                "cpT_X",
                -0.75,
                1.0,
                0.01,
                -0.3,
                0.4,
                focalPointXPlanes,
                viewPointXPlanes,
                0.1
        );
        planeExportWParams(activeSim, cpTXSceneParams, regions, true, false, colormap);
        
        
        SceneExportParameters cpTYSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                "cpT_Y",
                -0.75,
                1.0,
                0.01,
                -0.15,
                0.15,
                focalPointYPlanes,
                viewPointYPlanes,
                0.2
        );
        
        planeExportWParams(activeSim, cpTYSceneParams, regions, true, false, colormap);
        
        SceneExportParameters cpTZSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                "cpT_Z",
                -0.75,
                1.0,
                0.01,
                0,
                0.15,
                focalPointZPlanes,
                viewPointZPlanes,
                0.2
        );
        planeExportWParams(activeSim, cpTZSceneParams, regions, true, false, colormap);
        
        SceneExportParameters lb2ZSceneParams = new SceneExportParameters(
                "Lambda2",
                "Lambda2_Z",
                -200000,
                0,
                0.01,
                0,
                0.15,
                focalPointZPlanes,
                viewPointZPlanes,
                0.2
        );
        planeExportWParams(activeSim, lb2ZSceneParams, regions, true, false, colormap);
        

        SceneExportParameters lb2XSceneParams = new SceneExportParameters(
                "Lambda2",
                "Lambda2_X",
                -200000,
                0,
                0.01,
                -0.3,
                0.4,
                focalPointXPlanes,
                viewPointXPlanes,
                0.1
        );
        planeExportWParams(activeSim, lb2XSceneParams, regions, true, false, colormap);

        SceneExportParameters lb2YSceneParams = new SceneExportParameters(
                "Lambda2",
                "Lambda2_Y",
                -200000,
                1.0,
                0.01,
                -0.15,
                0.15,
                focalPointYPlanes,
                viewPointYPlanes,
                0.2
        );

        planeExportWParams(activeSim, lb2YSceneParams, regions, true, false, colormap);

        
        // exporting detailed plane sections based on coordinate system names
        detailPlaneExport(colormap);
        exportWallYplus(activeSim, carquery, keepVis);

        surfaceScalar(activeSim, carquery, true, FF_PRESSURECOEFFICIENT_NAME, -4.5, 1, keepVis, colormap);
        surfaceScalar(activeSim, carquery, false, FF_WALLSHEARSTRESS_NAME, 0, 1.7, keepVis, colormap);
        
        SceneExportParameters velXSceneParams = new SceneExportParameters(
                "Velocity",
                "vel_X",
                0,
                40,
                0.01,
                -0.3,
                0.4,
                focalPointXPlanes,
                viewPointXPlanes,
                0.1
        );
        planeExportWParams(activeSim, velXSceneParams, regions, false, false, colormap);
        
        SceneExportParameters velYSceneParams = new SceneExportParameters(
                "Velocity",
                "vel_Y",
                0,
                40,
                0.01,
                -0.15,
                0.15,
                focalPointYPlanes,
                viewPointYPlanes,
                0.2
        );
        planeExportWParams(activeSim, velYSceneParams, regions, false, true, colormap);

        
        SceneExportParameters velZSceneParams = new SceneExportParameters(
                "Velocity",
                "vel_Z",
                0,
                40,
                0.01,
                0,
                0.15,
                focalPointZPlanes,
                viewPointZPlanes,
                0.2
        );
        planeExportWParams(activeSim, velZSceneParams, regions, false, true, colormap);
        
    }


    private ArrayList<String> readFile(Simulation activeSim, String filePath, String filename) {
        ArrayList<String> filecontent = new ArrayList<>();

        // Single Monitors
        try (BufferedReader br = new BufferedReader(new FileReader(filePath + File.separator + filename))) {
            String line = br.readLine();

            while (line != null) {
                filecontent.add(line);
                activeSim.println(line);
                line = br.readLine();
            }
        } catch (FileNotFoundException ex) {
            activeSim.println("Report-averaging Init: File " + filename + " could not be created/found.\nError:\n" + ex.getMessage());
        } catch (IOException ex) {
            activeSim.println("An error occurred while reading " + filename + "ERROR:\n" + ex.getMessage());
        }
        return filecontent;
    }

    private void exportSpecPlot(Simulation activeSim, boolean keepVis, String plotname, String targetDir, double[] LA_Bounds, double[] BA_Bounds) {
        //start timing
        long startTime = System.currentTimeMillis();

        try {
            MonitorPlot monitorPlot = ((MonitorPlot) activeSim.getPlotManager().getPlot(plotname));

            monitorPlot.open();

            //monitorPlot.resetChartBounds();

            Cartesian2DAxisManager axisManager = ((Cartesian2DAxisManager) monitorPlot.getAxisManager());

            Cartesian2DAxis bottomAxis = ((Cartesian2DAxis) axisManager.getAxis("Bottom Axis"));

            if (BA_Bounds != null) {
                activeSim.println("applying BA_Bounds");
                bottomAxis.setMinimum(BA_Bounds[0]);
                bottomAxis.setMaximum(BA_Bounds[1]);
            } else {
                bottomAxis.setLockMinimum(false);
                bottomAxis.setLockMaximum(false);
            }

            Cartesian2DAxis leftAxis = ((Cartesian2DAxis) axisManager.getAxis("Left Axis"));

            if (LA_Bounds != null) {
                activeSim.println("applying LA_Bounds");
                leftAxis.setMinimum(LA_Bounds[0]);
                leftAxis.setMaximum(LA_Bounds[1]);
            } else {
                leftAxis.setLockMinimum(false);
                leftAxis.setLockMaximum(false);
            }

            monitorPlot.encode(resolvePath(targetDir + File.separator + plotname.replace(" ", "_") + ".png"), "png", 3000, 2250);

            if (!keepVis) {
                monitorPlot.close();
            }

        } catch (Exception e) {
            activeSim.println("!!!Error!!! Could not evaluate " + plotname + "\nError:\n" + e);
        }

        //End timing
        long endTime = System.currentTimeMillis();
        double timeTaken = ((double) (endTime - startTime)) / 1000;
        activeSim.println("Time taken exporting " + plotname + " plot: " + timeTaken + "s");
    }
    
    
    private void exportWallYplus(Simulation activeSim, Query carquery, boolean keepVis) {

        // TODO maybe want to add this scene in every case

        //start timing
        long startTime = System.currentTimeMillis();

        // Exportieren des Wall y+ Displayers

        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("exporting Wall Y+ images");

        String wallYpDir = postProDirStr + File.separator + "WallYPlus";
        File wallYpDirFile = new File(wallYpDir);

        if (wallYpDirFile.exists() && Objects.requireNonNull(wallYpDirFile.listFiles()).length >= 6) {
            activeSim.println("Wall Y+ images already exist. Skipping");
        } else {

            //creating DoubleVectors for POV
            DoubleVector focalpoint;
            DoubleVector pointOfView;
            DoubleVector viewUp;
            Double zoom;

            Scene wallYpScene = activeSim.getSceneManager().createScene();

            wallYpScene.initializeAndWait();

            ScalarDisplayer scalarDisplayer = wallYpScene.getDisplayerManager().createScalarDisplayer("Scalar");
            scalarDisplayer.initialize();

            PrimitiveFieldFunction wallYplusFF = ((PrimitiveFieldFunction) activeSim.getFieldFunctionManager().getFunction("WallYplus"));

            scalarDisplayer.getScalarDisplayQuantity().setFieldFunction(wallYplusFF);

            scalarDisplayer.getInputParts().setQuery(null);
            scalarDisplayer.getInputParts().setQuery(carquery);

            wallYpScene.open();

            CurrentView currentView = wallYpScene.getCurrentView();

            //exporting the bottom view
            focalpoint = new DoubleVector(new double[]{0, 0, 0.05});
            pointOfView = new DoubleVector(new double[]{0, 0.0, -1});
            viewUp = new DoubleVector(new double[]{0, -1, 0});
            zoom = 0.2;
            currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);

            //high yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{30.0, 300.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_high_bottom.png"), 1, 2000, 1500, true, false);

            //Buffer yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{5.0, 30.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_buffer_bottom.png"), 1, 2000, 1500, true, false);

            //low yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{0.0, 5.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_low_bottom.png"), 1, 2000, 1500, true, false);


            //exporting the iso view
            focalpoint = new DoubleVector(new double[]{0, 0, 0.07});
            pointOfView = new DoubleVector(new double[]{-1, -1, 1});
            viewUp = new DoubleVector(new double[]{0, 0, 1});
            zoom = 0.2;
            currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);

            //high yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{30.0, 300.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_high_iso.png"), 1, 2000, 1500, true, false);

            //Buffer yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{5.0, 30.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_buffer_iso.png"), 1, 2000, 1500, true, false);

            //low yplus

            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{0.0, 5.0}));
            wallYpScene.printAndWait(resolvePath(wallYpDir + File.separator + "_Wall_Y_Plus_low_iso.png"), 1, 2000, 1500, true, false);


            if (!keepVis) {
                activeSim.getSceneManager().deleteScene(wallYpScene);
            }

            activeSim.println("done exporting Wall Y+ images");

            //End timing
            long endTime = System.currentTimeMillis();
            double timeTaken = ((double) (endTime - startTime)) / 1000;

            activeSim.println("Time taken exporting Wall Y+ images: " + timeTaken + "s");
        }
    }
    

    public void averageReports(String versionNumber, int iterationsToAverage, List<String> monitorNames, Simulation sim) {

        //start timing
        long startTime = System.currentTimeMillis();
        
        List<String> allMonitorNames = new ArrayList();;
        allMonitorNames.addAll(monitorNames);
        sim.println(allMonitorNames);


        double[] reportMonitor_values = new double[iterationsToAverage];  // Zwischenspeicher fuer Monitor Werte in den Schleifen. Die Länge, die hier fetgelegt wird, ist im folgenden völlig egal
        double[] addierte_values = new double[allMonitorNames.size()]; // Summe der zu mittelnden Werte je Monitor (in jedem Platz des Arrays wird jeweils die Summe für den entsprechenden Monitor gespeichert)
        ReportMonitor zwischenReportMonitor; // Variable fuer die Schleifen in denen alle y-Werte zwischengespeichert werden.
    
    /*
        Die zwei Schleifen gehen alle Uebergebenen Monitore durch und summieren deren Werte jeweils auf.
        Gespeichert wird in addierteLiftDrag_values, in der uebergebenen Monitorname Reihenfolge
    */
        for (int i = 0; i < allMonitorNames.size(); i++) //i zählt alle Monitore in allMonitorNames durch
        {
            zwischenReportMonitor = ((ReportMonitor) sim.getMonitorManager().getMonitor(allMonitorNames.get(i))); //Erstellt das Monitor-Objekt
            reportMonitor_values = zwischenReportMonitor.getAllYValues(); //Achtung, das Array wird dadurch überschrieben und die Länge neu festgelegt
            addierte_values[i] = 0; // Auf 0 setzen. Eigentlich nicht noetig aber zur Sicherheit.

            for (int j = reportMonitor_values.length - (iterationsToAverage); j < reportMonitor_values.length; j++) //Durchläuft die Schleife für die letzten Iterationen der Anzahl iterationsToAverage
            {
                addierte_values[i] += reportMonitor_values[j]; //Summiert mit jedem Schritt den naechsten zu mittelnden Wert auf.
            }
        }
    
    
    
    /*
        In diesem Try Block der verschiedene IOException wirft wird die Ausgabe in eine .txt geschrieben
    */
        //File textAuswertung = new File(postProDirStr + File.separator + "Auswertung_" + versionNumber + ".txt");
        File textAuswertung = new File(sim.getSessionDir() + File.separator + ".." + File.separator + "Auswertung_" + versionNumber + ".txt");

        try (FileWriter fw = new FileWriter(textAuswertung, true); BufferedWriter bw = new BufferedWriter(fw)) {

            sim.println(textAuswertung);

            // Writes first line if file is empty.
            if (textAuswertung.length() == 0) {
                bw.write("Simulation\t");
                for (String monitorNameLiftDrag : allMonitorNames) {
                    bw.write(monitorNameLiftDrag + "\t");
                }
            }
            bw.newLine(); //Neue Zeile
            bw.write(sim.getPresentationName() + "\t");

            //Ausgabe zweite Zeile:
            for (int i = 0; i < addierte_values.length; i++) {
                //if (i >= single_lengh)
                //    addierte_values[i] *= 1; // Werte werden mal 2 genommen. Symmetriebedingung. TODO Bug?? values are not multipleid by 2
                addierte_values[i] = addierte_values[i] / iterationsToAverage; // Mittelung der Werte
                bw.write(addierte_values[i] + "\t");
            }
        } catch (FileNotFoundException ex) {
            System.err.println("The averaged-report file could not be created/found.\nError:\n" + ex.getMessage());
        } catch (IOException ex) {
            System.err.println("An error occurred while creating the averaged-reports file. See message for details.\nError:\n" + ex.getMessage());
        }

        //End timing
        long endTime = System.currentTimeMillis();
        double timeTaken = ((double) (endTime - startTime)) / 1000;

        sim.println("Time taken writing reports: " + timeTaken + "s");

    }


    private void planeExportWParams(Simulation activeSim, SceneExportParameters p, HashSet regions, boolean scalar, boolean vectorScene, String colormap) {

        ScalarDisplayer scalarDisplayer = null;
        VectorDisplayer vectorDisplayer = null;
        /*
        Diese Methode exportiert eine Serie von Bildern einer schrittweise verschobenen Schnittebene mit einer Skalaren Funktion.
        Die Blickrichtung ist dabei immer senkrecht zur Ebene. Die Verschiebung der Ebene findet ebenfalls in dieser Richtung statt.
        Schrittweite, Start- und Endwert werden über Parameter vorgegeben.

        This method exports a series of images of a plane section with a scalar displayer.
        The POV and viewing direction are defined as parameters. The created sections are always perpendicular to the viewing direction.
        Also, the offset shift of the plane is always done in this direction.
        Offset coordinates and step size for shift offset are defined as parameters.
        Since the plane is created at the coordinate origin, all offsets refer to this origin.
        */

        //start timing
        long startTime = System.currentTimeMillis();

        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("exporting sections for:  " + p.fieldFunction + " | " + p.view_name);

        String targetDirectory = postProDirStr + File.separator + p.view_name + File.separator;
        File targetDirectoryFile = new File(targetDirectory);

        if (targetDirectoryFile.exists() && Objects.requireNonNull(targetDirectoryFile.listFiles()).length > 10) {
            activeSim.println("Directory for " + p.fieldFunction + " | " + p.view_name + " already exists and contains more than 10 files. Skipping. Delete the directory if it should be created new");
        } else {

            //calculating viewing direction from difference of focalpoint and point of view
            DoubleVector ViewDirection = new DoubleVector(new double[p.pointOfView.size()]);
            for (int i = 0; i < p.pointOfView.size(); i++) {
                ViewDirection.setComponent(i, p.focalpoint.getComponent(i) - p.pointOfView.getComponent(i));
            }

            DoubleVector ViewDirectionNormalized = normalize(ViewDirection); //normalizing the view direction vector to be able to scale it later on.
            activeSim.println("Viewing direction: " + ViewDirection.toString());
            activeSim.println("Normalized viewing direction: " + ViewDirectionNormalized.toString());


            //Creating the plane section in model origin with
            PlaneSection planeSection = (PlaneSection) activeSim.getPartManager().createImplicitPart(new NeoObjectVector(new Object[]{}), ViewDirection, new DoubleVector(new double[]{0.0, 0.0, 0.0}), 0, 1, new DoubleVector(new double[]{0.0}));

            planeSection.setPresentationName("plane section");

            //defining all regions and boundaries and assinging them to the plane section
            planeSection.getInputParts().setObjects(regions);

            //creating the scene
            Scene scene = activeSim.getSceneManager().createScene();

            scene.initializeAndWait();

            //scene settings
            scene.setBackgroundColorMode(BackgroundColorMode.SOLID);

            SolidBackgroundColor backgroundColor = scene.getSolidBackgroundColor();

            backgroundColor.setColor(new DoubleVector(new double[]{0.4666999876499176, 0.53329998254776, 0.6000000238418579}));

            //importing the fieldFunction
            FieldFunction fieldFunctionToEval = (activeSim.getFieldFunctionManager().getFunction(p.fieldFunction));

            // change to velocity in rotating air if cornering case is chosen
            if (isCornering && p.fieldFunction.equals("Velocity")) {
                UserRotatingReferenceFrame rotatingAirRF = ((UserRotatingReferenceFrame) activeSim.get(ReferenceFrameManager.class).getObject(ROTATING_AIR_RF_NAME));
                fieldFunctionToEval = fieldFunctionToEval.getFunctionInReferenceFrame(rotatingAirRF);
            }

            Legend legend;
            //creating a scalar displayer
            if (vectorScene) {
                vectorDisplayer = scene.getDisplayerManager().createVectorDisplayer("Vector");
                vectorDisplayer.setDisplayMode(VectorDisplayMode.VECTOR_DISPLAY_MODE_LIC);
                //determining if half car sim is used, and mirroring the solution
                /*
                try {
                    if (ishalfCar(activeSim)) {
                        activeSim.println("Next up: applying mirror on Displayer");
                        GraphicsSymmetricRepeat graphicsSymmetricRepeat_halfcar = ((GraphicsSymmetricRepeat) activeSim.getTransformManager().getObject("HalfCar_Sym_Transform"));
                        vectorDisplayer.setVisTransform(graphicsSymmetricRepeat_halfcar);
                        activeSim.println("applied mirror Transform on Displayer");
                    }
                } catch (Exception e) {
                    activeSim.println("Some Error occurred, while applying mirror on Displayers\nErrormessage:\n" + e);
                }
                */
            
                legend = vectorDisplayer.getLegend();

                vectorDisplayer.initialize();
                vectorDisplayer.getInputParts().setQuery(null);
                vectorDisplayer.getInputParts().setObjects(planeSection);

                vectorDisplayer.getVectorDisplayQuantity().setClip(ClipMode.NONE);
                vectorDisplayer.getVectorDisplayQuantity().setFieldFunction(fieldFunctionToEval);

                // increase stepSize and reduce number of integration steps to improve performance a bit
                vectorDisplayer.getLICSettings().setStepSize(3.0);
                vectorDisplayer.getLICSettings().setNumberOfSteps(2);
                vectorDisplayer.getLICSettings().setIntensity(0.45);

                vectorDisplayer.getVectorDisplayQuantity().setRange(new DoubleVector(new double[]{p.scaleMin, p.scaleMax}));

            } else {
                //creating a scalar displayer
                scalarDisplayer = scene.getDisplayerManager().createScalarDisplayer("Scalar");

                //assigning the plane section to the scalar displayer
                scalarDisplayer.initialize();
                scalarDisplayer.getInputParts().setQuery(null);
                scalarDisplayer.getInputParts().setObjects(planeSection);

                //setting the displayer properties
                scalarDisplayer.setFillMode(ScalarFillMode.NODE_FILLED);
                scalarDisplayer.getScalarDisplayQuantity().setClip(ClipMode.NONE);

                legend = scalarDisplayer.getLegend();

                //checking if field function is a vector and using the magnitude if true.
                if (scalar) {
                    // assigning fieldFunction to displayer
                    scalarDisplayer.getScalarDisplayQuantity().setFieldFunction(fieldFunctionToEval);
                } else {
                    try {
                        VectorMagnitudeFieldFunction vectorFF = ((VectorMagnitudeFieldFunction) fieldFunctionToEval.getMagnitudeFunction());
                        // assigning fieldFunction to displayer
                        scalarDisplayer.getScalarDisplayQuantity().setFieldFunction(vectorFF);
                    } catch (Exception e) {
                        activeSim.println("---------------------------------------------------------------------------");
                        activeSim.println("!!! Error. Fieldfunction might not be a scalar. Cant convert to magnitude. Aborting creation of plane sections. \nError:\n" + e);
                        activeSim.println("---------------------------------------------------------------------------");
                    }
                }

                // set the scale
                scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{p.scaleMin, p.scaleMax}));
            }

            //legend settings
            if (colormap.equals("spectrum")){
                SpectrumLookupTable spectrumLookupTable_0 =((SpectrumLookupTable) activeSim.get(LookupTableManager.class).getObject("spectrum"));
                legend.setLookupTable(spectrumLookupTable_0);
                activeSim.println("Using spectrum Colormap now.");
            }else{
                if(!colormap.equals("thermal")){
                    activeSim.println("no valid colormap spcified! ONLY spectrum or thermal are supported!");
                    activeSim.println("Using thermal Colormap now.");
                }
                PredefinedLookupTable thermalLookupTable = ((PredefinedLookupTable) activeSim.get(LookupTableManager.class).getObject("thermal"));
                legend.setLookupTable(thermalLookupTable);
            }
            legend.setLevels(64);
            legend.setHeight(0.01);
            legend.setTitleHeight(0.024);
            legend.setLabelHeight(0.018);

            //setting the view according to input of focalpoint, pointOfView and zoom. Z depends on name
            CurrentView view = scene.getCurrentView();

            // TODO this is a more than ugly workaround. Better idea would be to get all the view dependant parameters down in this method and only
            // TODO specify which direction should be evaluated
            if (p.view_name.endsWith("_Z")) {
                // negative y direction should be upside of the image
                view.setInput(p.focalpoint, p.pointOfView, new DoubleVector(new double[]{rotateCoordSystem.getBasis1().getComponent(0), rotateCoordSystem.getBasis1().getComponent(1), 0}), p.zoom, 1, 1.0);
                
            } else {
                view.setInput(p.focalpoint, p.pointOfView, new DoubleVector(new double[]{0.0, 0.0, 1.0}), p.zoom, 1, 1.0);
            }

            int counter = 1; //counter for file names to sort the files by their names in correct order


            //in the following loop the plane origin is set off to "a", which ranges from the user defined start to end Coordinates and increases by step size.
            //for each position an image is exported
            // TODO make this independant of viewdirection


            for (double coordinate = p.startCoordinate; coordinate < p.endCoordinate; coordinate = coordinate + p.stepSize) {

                String counterString = addLeadingZeros(counter, 3); //adding leading zeros to the file counter to make the windows explorer sort the files in the correct order.

                //set the origin of the plane section to the pointOfViewNormalized vector scaled with "a".
                DoubleVector planeOrigin;
                // TODO this is a super ugly workaround here
                double x_origin = ViewDirectionNormalized.getComponent(0) * coordinate;
                double y_origin = ViewDirectionNormalized.getComponent(1) * coordinate;
                //if halfcar sim, take the plane cut from the left vehicle side


                double z_origin = ViewDirectionNormalized.getComponent(2) * coordinate;
                if (p.view_name.endsWith("_Z")) {
                    z_origin = ViewDirectionNormalized.getComponent(2) * (-1) * coordinate;
                }
                planeOrigin = new DoubleVector(new double[]{x_origin, y_origin, z_origin});
                planeSection.setOrigin(planeOrigin);

                //printing the scene to an image file
                scene.printAndWait(resolvePath(targetDirectory + File.separator + p.view_name + counterString + "PlaneOffset" + Math.round(coordinate * 100) / 100.0 + ".png"), 1, 1800, 1200, true, false);
                counter++;
            }

            //deleting created scene and plane
            if (!keepVis) {
                activeSim.getPartManager().deletePart(planeSection);
                activeSim.getSceneManager().deleteScene(scene);
            }
            activeSim.println("done exporting planes for " + p.fieldFunction + " | " + p.view_name);

            //End timing
            long endTime = System.currentTimeMillis();
            double timeTaken = ((double) (endTime - startTime)) / 1000;

            activeSim.println("Time taken exporting planes: " + timeTaken + "s");
        }
    }

    
    private void detailPlaneExport(String colormap) {

        Collection<Object> allCoordinateSystems = activeSim.getCoordinateSystemManager().getLabCoordinateSystem().getLocalCoordinateSystemManager().getChildren();
        allCoordinateSystems.addAll(rotateCoordSystem.getLocalCoordinateSystemManager().getChildren());

        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("Exporting detailed plane sections.");
        for (Object csObj : allCoordinateSystems) {

            if ((csObj instanceof CartesianCoordinateSystem)) {


                CoordinateSystem cs = (CartesianCoordinateSystem) csObj;
                String csName = cs.getQualifiedName();
                // TODO this is an ugly workaround. dont know how to get rid of this "Laboratory->" Prefix otherwise
                csName = csName.replace("Laboratory->", "");

                // have to use correct naming conventions to get found
                if ((csName.toUpperCase().startsWith("EVAL__") || csName.toUpperCase().startsWith("EVALSOL__")) && csName.toUpperCase().endsWith("__ON")) {

                    activeSim.println("Now Evaluating " + csName);
                    String csViewname = csName.replaceAll("(?i)__ON", "").replaceAll("(?i)EvalSOL__", "").replaceAll("(?i)Eval__", "");
                    createDetailedPlanes(cs.getOriginVector(), csViewname, colormap);
                }
            }
        }
        activeSim.println("Exporting detailed plane sections from simConfig parameters.");
        printDetailedPlanesFromConfig(colormap);
    }
    

    private String readColormap(){
        //reads colormap parameter from simconfig, if it exists
        //initialization
        String COLORMAP="spectrum";
        // TODO this is a duplicate with code from ChangeParameters.java.
        String sessionDir = activeSim.getSessionDir();

        return COLORMAP;
    }


    private void printDetailedPlanesFromConfig(String colormap) {

        // TODO this is a duplicate with code from ChangeParameters.java.
        String sessionDir = activeSim.getSessionDir();

        File simConfigFile = new File(sessionDir + File.separator + SIMCONFIG_FILE_NAME);

        if (simConfigFile.exists()) {

            try (BufferedReader br = new BufferedReader(new FileReader(simConfigFile.getAbsolutePath()))) {
                String line;
                // TODO what happens if there is no value behind the '=' sign. should be failsafe
                while ((line = br.readLine()) != null) {
                    line = line.trim();

                    if (line.toLowerCase().startsWith(DETAILED_PLANE_EXPORTS_KEY.toLowerCase() + KEY_VALUE_DELIMITER)) {
                        String planeParameters = line.substring(line.indexOf(KEY_VALUE_DELIMITER) + 1);
                        String[] planeParametersArray = planeParameters.split(PARAMETER_DELIMITER);

                        for (String s : planeParametersArray) {
                            String name = s.substring(0, s.indexOf(KEY_VALUE_DELIMITER));

                            if ((name.toUpperCase().startsWith("EVAL__") || name.toUpperCase().startsWith("EVALSOL__")) && name.toUpperCase().endsWith("__ON")) {
                                name = name.replaceAll("(?i)__ON", "").replaceAll("(?i)EvalSol__", "").replaceAll("(?i)Eval__", "");
                                String origin = s.substring(s.indexOf(KEY_VALUE_DELIMITER) + 1);
                                String[] originSplit = origin.split(VALUE_DELIMITER);
                                originSplit[0] = originSplit[0].replace("[", "");
                                originSplit[2] = originSplit[2].replace("]", "");

                                DoubleVector originVector = new DoubleVector(new double[]{Double.parseDouble(originSplit[0]), Double.parseDouble(originSplit[1]), Double.parseDouble(originSplit[2])});
                                createDetailedPlanes(originVector, name, colormap);
                            }
                        }
                    }
                }
            } catch (Exception e) {
                activeSim.println("!!! Error: Some error occurred while reading in the parameters from the simConfig.txt.\nErrormessage:\n" + e);
            }
        }
    }


    // TODO this is a very ugly duplicate fom the mesh_eval_script and the CellQualityScences script
    private DoubleVector rotateOrigin(DoubleVector point) {
        //This Method provides a transformation from the para_rotate coordinate system to the global coordinate system (rotation & translation)
        //get origin of para rotate system to apply offset later on
        DoubleVector paraRotateOrigin = rotateCoordSystem.getOriginVector();

        // defining basis vectors for the coordinate transformation through Basis-Matrix Multiplication
        DoubleVector rotBasis0 = rotateCoordSystem.getBasis0();
        DoubleVector rotBasis1 = rotateCoordSystem.getBasis1();
        DoubleVector rotBasis2 = rotateCoordSystem.getBasis2();
        // defining less wordy point coords (in rotate basis)
        double x_rot = point.getComponent(0);
        double y_rot = point.getComponent(1);
        double z_rot = point.getComponent(2);

        // coordinate transformation: Point_global = A_rot_basis_matrix * Point_rot + Rot_origin_global
        // step1: rotate coordinates into global basis directions A_rot_basis_matrix= [rot_basis_vec_0 , rot_basis_vec_1 , rot_basis_vec_2]
        double x_temp = rotBasis0.getComponent(0) * x_rot + rotBasis1.getComponent(0) * y_rot + rotBasis2.getComponent(0) * z_rot;
        double y_temp = rotBasis0.getComponent(1) * x_rot + rotBasis1.getComponent(1) * y_rot + rotBasis2.getComponent(1) * z_rot;
        double z_temp = rotBasis0.getComponent(2) * x_rot + rotBasis1.getComponent(2) * y_rot + rotBasis2.getComponent(2) * z_rot;

        // step2: apply offset to allow for translation of origin coord Point= Point_temp + Rot_origin_global
        double x = x_temp + paraRotateOrigin.getComponent(0);
        double y = y_temp + paraRotateOrigin.getComponent(1);
        double z = z_temp + paraRotateOrigin.getComponent(2);

        return new DoubleVector(new double[]{x, y, z});
    }


    private void createDetailedPlanes(DoubleVector origin, String csViewname, String colormap) {


        // defining viewpoint in global coord. system based on rot values with transformation
        DoubleVector viewPointXPlanes = rotateOrigin(new DoubleVector(new double[]{origin.getComponent(0) - 1, origin.getComponent(1), origin.getComponent(2)}));
        DoubleVector viewPointYPlanes = rotateOrigin(new DoubleVector(new double[]{origin.getComponent(0), origin.getComponent(1) - 1, origin.getComponent(2)}));
        DoubleVector viewPointZPlanes = rotateOrigin(new DoubleVector(new double[]{origin.getComponent(0), origin.getComponent(1), origin.getComponent(2) + 1}));

        // converting focal point to global coord
        origin = rotateOrigin(origin);


        double delta = 0.2;
        int numPlanes = 20;
        double stepSize = 2 * delta / numPlanes;

        SceneExportParameters cpTXSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                csViewname + "_cpT_X",
                -0.75,
                1,
                stepSize,
                origin.getComponent(0) - delta,
                origin.getComponent(0) + delta,
                origin,
                viewPointXPlanes,
                0.3
        );
        planeExportWParams(activeSim, cpTXSceneParams, regions, true, false,colormap);

        SceneExportParameters cpTYSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                csViewname + "_cpT_Y",
                -0.75,
                1,
                stepSize,
                origin.getComponent(1) - delta,
                origin.getComponent(1) + delta,
                origin,
                viewPointYPlanes,
                0.3
        );
        planeExportWParams(activeSim, cpTYSceneParams, regions, true, false, colormap);

        SceneExportParameters cpTZSceneParams = new SceneExportParameters(
                FF_CPT_NAME,
                csViewname + "_cpT_Z",
                -0.75,
                1,
                stepSize,
                origin.getComponent(2) - delta,
                origin.getComponent(2) + delta,
                origin,
                viewPointZPlanes,
                0.3
        );
        planeExportWParams(activeSim, cpTZSceneParams, regions, true, false, colormap);

        SceneExportParameters velXSceneParams = new SceneExportParameters(
                "Velocity",
                csViewname + "_vel_X",
                0,
                40,
                stepSize,
                origin.getComponent(0) - delta,
                origin.getComponent(0) + delta,
                origin,
                viewPointXPlanes,
                0.3
        );
        planeExportWParams(activeSim, velXSceneParams, regions, false, false, colormap);

        SceneExportParameters velYSceneParams = new SceneExportParameters(
                "Velocity",
                csViewname + "_vel_Y",
                0,
                40,
                stepSize,
                origin.getComponent(1) - delta,
                origin.getComponent(1) + delta,
                origin,
                viewPointYPlanes,
                0.3
        );
        planeExportWParams(activeSim, velYSceneParams, regions, false, true, colormap);

        SceneExportParameters velZSceneParams = new SceneExportParameters(
                "Velocity",
                csViewname + "_vel_Z",
                0,
                40,
                stepSize,
                origin.getComponent(2) - delta,
                origin.getComponent(2) + delta,
                origin,
                viewPointZPlanes,
                0.3
        );
        planeExportWParams(activeSim, velZSceneParams, regions, false, true, colormap);

    }

    // TODO what is this used for? Seems unused
    public void resampledVolumeScene(Simulation activeSim, HashSet regions, Query carquery, String fieldFunction, double scaleMin, double scaleMax, boolean keepVis) {

        //start timing
        long startTime = System.currentTimeMillis();

        activeSim.println("exporting Resampled Volume");

        Scene resampledVolumeScene = activeSim.getSceneManager().createScene();

        //Scene Settings
        resampledVolumeScene.setBackgroundColorMode(BackgroundColorMode.SOLID);

        SolidBackgroundColor backgroundCol =
                resampledVolumeScene.getSolidBackgroundColor();

        backgroundCol.setColor(new DoubleVector(new double[]{1, 1, 1}));

        resampledVolumeScene.initializeAndWait();

        PartDisplayer geometryDisp = resampledVolumeScene.getDisplayerManager().createPartDisplayer("Geometry", -1, 4);
        geometryDisp.initialize();
        geometryDisp.getInputParts().setQuery(null);
        geometryDisp.getInputParts().setQuery(carquery);
        geometryDisp.setSurface(true);
        geometryDisp.setOutline(false);

        /*
        //determining if half car sim is used, and mirroring the solution
        try {
            if (ishalfCar(activeSim)) {
                activeSim.println("Next up: applying mirror on Displayer");
                GraphicsSymmetricRepeat graphicsSymmetricRepeat_halfcar = ((GraphicsSymmetricRepeat) activeSim.getTransformManager().getObject("HalfCar_Sym_Transform"));
                geometryDisp.setVisTransform(graphicsSymmetricRepeat_halfcar);
                activeSim.println("applied mirror Transform on Displayer");
            }
        } catch (Exception e) {
            activeSim.println("Some Error occurred, while applying mirror on Displayers\nErrormessage:\n" + e);
        }
        */
        Units units_1 = activeSim.getUnitsManager().getPreferredUnits(new IntVector(new int[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        Units units_0 = activeSim.getUnitsManager().getPreferredUnits(new IntVector(new int[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

        PartDisplayer partDisp = ((PartDisplayer) resampledVolumeScene.getCreatorDisplayer());
        //determining if half car sim is used, and mirroring the solution
        try {
            if (ishalfCar(activeSim)) {
                activeSim.println("Next up: applying mirror on Displayer");
                GraphicsSymmetricRepeat graphicsSymmetricRepeat_halfcar = ((GraphicsSymmetricRepeat) activeSim.getTransformManager().getObject("HalfCar_Sym_Transform"));
                partDisp.setVisTransform(graphicsSymmetricRepeat_halfcar);
                activeSim.println("applied mirror Transform on Displayer");
            }
        } catch (Exception e) {
            activeSim.println("Some Error occurred, while applying mirror on Displayers\nErrormessage:\n" + e);
        }

        partDisp.initialize();
        resampledVolumeScene.getCreatorGroup().setQuery(null);
        resampledVolumeScene.getCreatorGroup().setObjects(regions);
        ScalarDisplayer ffscalarDisp = resampledVolumeScene.getDisplayerManager().createScalarDisplayer("Resampled Volume Scalar");

        /*
        //determining if half car sim is used, and mirroring the solution
        try {
            if (ishalfCar(activeSim)) {
                activeSim.println("Next up: applying mirror on Displayer");
                GraphicsSymmetricRepeat graphicsSymmetricRepeat_halfcar = ((GraphicsSymmetricRepeat) activeSim.getTransformManager().getObject("HalfCar_Sym_Transform"));
                ffscalarDisp.setVisTransform(graphicsSymmetricRepeat_halfcar);
                activeSim.println("applied mirror Transform on Displayer");
            }
        } catch (Exception e) {
            activeSim.println("Some Error occurred, while applying mirror on Displayers\nErrormessage:\n" + e);
        }
        */
        /*
        Legend legend_0 = ffscalarDisp.getLegend();
        SpectrumLookupTable spectrumLookupTable_0 = ((SpectrumLookupTable) activeSim.get(LookupTableManager.class).getObject("spectrum"));
        legend_0.setLookupTable(spectrumLookupTable_0);        
        */
	/*
        legend legend_0 = ffscalarDisp.getLegend();
        PredefinedLookupTable thermalLookupTable_0 = ((PredefinedLookupTable) activeSim.get(LookupTableManager.class).getObject("thermal"));
        legend_0.setLookupTable(thermalLookupTable_0);
        ffscalarDisp.initialize();
	*/

        activeSim.println("check1");
        ResampledVolumePart resampledVolumePart = activeSim.getPartManager().createResampledVolumePart(new NeoObjectVector(regions.toArray()), 0.1, units_0, 0.004, units_0, units_0, units_0, new DoubleVector(new double[]{2.5663284580761103, -0.5489359277335215, 0.5268039417642574}), units_0, units_0, units_0, new DoubleVector(new double[]{6.781206952869266, 1.5888072779891433, 1.9216290123912447}), units_0, units_0, units_0, new DoubleVector(new double[]{0.0, 0.0, 1.0}), units_1, 0.0);
        activeSim.println("check2");
        ffscalarDisp.getVisibleParts().addParts(resampledVolumePart);
        resampledVolumeScene.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);
        ffscalarDisp.getScalarDisplayQuantity().setClip(ClipMode.MAX);
        PrimitiveFieldFunction primFieldFunction = ((PrimitiveFieldFunction) activeSim.getFieldFunctionManager().getFunction(fieldFunction));

        ffscalarDisp.getScalarDisplayQuantity().setFieldFunction(primFieldFunction);

        ffscalarDisp.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{scaleMin, scaleMax}));

        exportSdtViews(resampledVolumeScene, activeSim.getSessionPath() + fieldFunction + File.separator + "ResampledVolume_" + fieldFunction);
        resampledVolumeScene.export3DSceneFileAndWait(resolvePath(activeSim.getSessionPath() + fieldFunction + File.separator + "ResampledVolume_" + fieldFunction + "Scene" + ".sce"), "ResampledVolume_" + fieldFunction, "", false, false);

        //deleting created scene and RSV
        if (!keepVis) {
            activeSim.getPartManager().deletePart(resampledVolumePart);
            activeSim.getSceneManager().deleteScene(resampledVolumeScene);
        }
        activeSim.println("done exporting Resampled Volume");

        //End timing
        long endTime = System.currentTimeMillis();
        double timeTaken = ((double) (endTime - startTime)) / 1000;

        activeSim.println("Time taken exporting Resampled Volume: " + timeTaken + "s");
    }

    private void surfaceScalar(Simulation activeSim, Query carquery, boolean scalar, String fieldFunction, double scaleMin, double scaleMax, boolean keepVis, String colormap) {

        //start timing
        long startTime = System.currentTimeMillis();

        activeSim.println("---------------------------------------------------------------------------");
        activeSim.println("exporting surface Scalar: " + fieldFunction);

        String targetDirectory = postProDirStr + File.separator + "Surf_" + fieldFunction;
        File targetDirectoryFile = new File(targetDirectory);

        if (targetDirectoryFile.exists() && Objects.requireNonNull(targetDirectoryFile.listFiles()).length >= 6) {
            activeSim.println("Surface scalar plots for " + fieldFunction + " are already exported. Skipping. If you want to export again, delete the directory");
        } else {
            //creating the scene
            Scene scalarScene = activeSim.getSceneManager().createScene();

            scalarScene.initializeAndWait();

            //creating a scalar displayer
            ScalarDisplayer scalarDisplayer = scalarScene.getDisplayerManager().createScalarDisplayer("Scalar");

            //assigning the plane section to the scalar displayer
            scalarDisplayer.initialize();
            scalarDisplayer.getInputParts().setQuery(null);
            scalarDisplayer.getInputParts().setQuery(carquery);

            //determining if half car sim is used, and mirroring the solution
            try {
                if (ishalfCar(activeSim)) {
                    activeSim.println("Next up: applying mirror on Displayer");
                    GraphicsSymmetricRepeat graphicsSymmetricRepeat_halfcar = ((GraphicsSymmetricRepeat) activeSim.getTransformManager().getObject("HalfCar_Sym_Transform"));
                    scalarDisplayer.setVisTransform(graphicsSymmetricRepeat_halfcar);
                    activeSim.println("applied mirror Transform on Displayer");
                }
            } catch (Exception e) {
                activeSim.println("Some Error occurred, while applying mirror on Displayers\nErrormessage:\n" + e);
            }

            //importing the fieldFunction
            PrimitiveFieldFunction primitiveFF = ((PrimitiveFieldFunction) activeSim.getFieldFunctionManager().getFunction(fieldFunction));
            

            //checking if field function is a vector and using the magnitude if true.
            if (scalar) {
                // assigning fieldFunction to displayer
                scalarDisplayer.getScalarDisplayQuantity().setFieldFunction(primitiveFF);
                activeSim.println("changing legend to specified legend");
                Legend legend = scalarDisplayer.getLegend();
                //legend settings
                if (colormap.equals("spectrum")){
                    SpectrumLookupTable spectrumLookupTable_0 =((SpectrumLookupTable) activeSim.get(LookupTableManager.class).getObject("spectrum"));
                    legend.setLookupTable(spectrumLookupTable_0);
                    activeSim.println("Using spectrum Colormap now.");
                }else{
                    if(!colormap.equals("thermal")){
                        activeSim.println("no valid colormap spcified! ONLY spectrum or thermal are supported!");
                        activeSim.println("Using thermal Colormap now.");
                    }
                    PredefinedLookupTable thermalLookupTable = ((PredefinedLookupTable) activeSim.get(LookupTableManager.class).getObject("thermal"));
                    legend.setLookupTable(thermalLookupTable);
                }
            } else {
                VectorMagnitudeFieldFunction magnitude = ((VectorMagnitudeFieldFunction) primitiveFF.getMagnitudeFunction());
                // assigning fieldFunction to displayer
                scalarDisplayer.getScalarDisplayQuantity().setFieldFunction(magnitude);
            }


            //assigning fieldFunction to displayer and set the scale
            scalarDisplayer.getScalarDisplayQuantity().setRange(new DoubleVector(new double[]{scaleMin, scaleMax}));

            //setting the displayer properties
            scalarDisplayer.setFillMode(ScalarFillMode.NODE_FILLED);
            scalarDisplayer.getScalarDisplayQuantity().setClip(ClipMode.NONE);

            exportSdtViews(scalarScene, targetDirectory + File.separator + fieldFunction);

            //String sceneDirectoryString = postProDirStr + File.separator + "Scenes";
            String sceneDirectoryString = postProDirStr;
            File sceneDirectory = new File(sceneDirectoryString);
            if (!sceneDirectory.exists()) {
                if (!sceneDirectory.mkdirs()) {
                    activeSim.println("Error creating scene directory. Exporting scene might fail");
                }
            }
            //exporting scene
            // TODO should this always be exported? Maybe add an parameter and do this only in case
            scalarScene.export3DSceneFileAndWait(resolvePath(sceneDirectoryString + File.separator + "Surf_" + fieldFunction + "_scene.sce"), "SurfaceScalar_" + fieldFunction, "", false, false);

            //deleting created scene
            if (!keepVis) {
                activeSim.getSceneManager().deleteScene(scalarScene);
            }
            activeSim.println("done exporting surface Scalar");

            //End timing
            long endTime = System.currentTimeMillis();
            double timeTaken = ((double) (endTime - startTime)) / 1000;

            activeSim.println("Time taken exporting surface Scalar: " + timeTaken + "s");
        }
    }

    private static String addLeadingZeros(int number, int digits) {

        /*
        This method adds leading zeros to an integer 'number' and returns it as String with the length of 'digits'.
        Useful as counter in the beginning of file names.
        */

        StringBuilder zeros = new StringBuilder();
        for (int n = 1; n <= (digits - String.valueOf(number).length()); n++) {
            zeros.append("0");
        }
        return zeros.toString() + number;
    }

    private void exportSdtViews(Scene scene, String path) {
        
        /*
        this method exports .png files of a given Scene in standard views to a path.
        */

        CurrentView currentView = scene.getCurrentView();

        //creating DoubleVectors for POV
        DoubleVector focalpoint;
        DoubleVector pointOfView;
        DoubleVector viewUp;
        double zoom;

        //exporting the front view
        focalpoint = new DoubleVector(new double[]{0, 0.0, 0.05});
        pointOfView = new DoubleVector(new double[]{-1, 0.0, 0.05});
        viewUp = new DoubleVector(new double[]{0, 0, 1});
        zoom = 0.1;
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);

        scene.printAndWait(resolvePath(path + "_front.png"), 1, 2000, 1500, true, false);

        //exporting the back view
        focalpoint = new DoubleVector(new double[]{0, 0.0, 0.05});
        pointOfView = new DoubleVector(new double[]{1, 0.0, 0.05});
        viewUp = new DoubleVector(new double[]{0, 0, 1});
        zoom = 0.1;
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);
        scene.printAndWait(resolvePath(path + "_back.png"), 1, 2000, 1500, true, false);

        //exporting the side view
        focalpoint = new DoubleVector(new double[]{0, 0, 0.05});
        pointOfView = new DoubleVector(new double[]{0, -1, 0.05});
        viewUp = new DoubleVector(new double[]{0, 0, 1});
        zoom = 0.2;
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);
        scene.printAndWait(resolvePath(path + "_side.png"), 1, 2000, 1500, true, false);

        //exporting the top view
        focalpoint = new DoubleVector(new double[]{0, 0, 0.05});
        pointOfView = new DoubleVector(new double[]{0, 0.0, 1});
        viewUp = new DoubleVector(new double[]{0, 1, 0});
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);

        scene.printAndWait(resolvePath(path + "_top.png"), 1, 2000, 1500, true, false);

        //exporting the bottom view
        focalpoint = new DoubleVector(new double[]{0, 0, 0.05});
        pointOfView = new DoubleVector(new double[]{0, 0.0, -1});
        viewUp = new DoubleVector(new double[]{0, -1, 0});
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);

        scene.printAndWait(resolvePath(path + "_bottom.png"), 1, 2000, 1500, true, false);

        //exporting the iso view
        focalpoint = new DoubleVector(new double[]{0, 0, 0.07});
        pointOfView = new DoubleVector(new double[]{-1, -1, 1});
        viewUp = new DoubleVector(new double[]{0, 0, 1});
        zoom = 0.2;
        currentView.setInput(rotateOrigin(focalpoint), rotateOrigin(pointOfView), rotateOrigin(viewUp), zoom, 1, 1.0);
        scene.printAndWait(resolvePath(path + "_iso.png"), 1, 2000, 1500, true, false);
    }

    private static DoubleVector normalize(DoubleVector V) {
        
        /*
        this method normalizes DoubleVector V
        */

        //determining the length of V
        double vLength = 0;
        for (int i = 0; i < V.size(); i++) {
            vLength = vLength + Math.pow(V.getComponent(i), 2);
        }
        vLength = Math.sqrt(vLength);

        //normalizing Viewdirection
        DoubleVector VNormalized = new DoubleVector(new double[V.size()]);
        for (int i = 0; i < V.size(); i++) {
            VNormalized.setComponent(i, V.getComponent(i) / vLength);
        }
        return VNormalized;
    }


    private Double REF() {
        Simulation simulation = getActiveSimulation();
        double refvelocity = 1.0;
        String sessionpath = simulation.getSessionPath();

        if (sessionpath.contains("BP01")) {
            refvelocity = 14.0;
        } else if (sessionpath.contains("BP02")) {
            refvelocity = 12.0;
        } else if (sessionpath.contains("BP03")) {
            refvelocity = 11.0;
        } else if (sessionpath.contains("BP04")) {
            refvelocity = 20.0;
        } else if (sessionpath.contains("BP05")) {
            refvelocity = 26.0;
        } else if (sessionpath.contains("BP06")) {
            refvelocity = 19.0;
        } else if (sessionpath.contains("BP07")) {
            refvelocity = 19.0;
        }
        return refvelocity;
    }


    private boolean ishalfCar(Simulation activeSim) {
        //this Method checks if the simulation state is halfCar and returns a true or false value, the default is false, in case an error appears
        try {
            ScalarGlobalParameter curr_simstate = ((ScalarGlobalParameter) activeSim.get(GlobalParameterManager.class).getObject("90_simstate"));
            if (curr_simstate.getQuantity().getRawValue() < 0.75) {
                activeSim.println("Simulation is a halfCar Sim");
                return true;
            } else {
                return false;
            }
        } catch (Exception e) {
            activeSim.println("!!! Error: Some error occurred while reading the Simstate parameter from .sim file and deciding it's state \nErrormessage:\n" + e);
            return false;
        }
    }
}


class SceneExportParameters {

    // class that holds all parameters that are used for scene exports

    // do these things public so that no getters and setters are needed.
    public String fieldFunction;
    public String view_name;
    public double scaleMin;
    public double scaleMax;
    public double stepSize;
    public double startCoordinate;
    public double endCoordinate;
    public DoubleVector focalpoint;
    public DoubleVector pointOfView;
    public double zoom;

    public SceneExportParameters(String fieldFunction, String view_name, double scaleMin, double scaleMax,
                                 double stepSize, double startCoordinate, double endCoordinate, DoubleVector focalpoint, DoubleVector pointOfView, double zoom) {
        this.fieldFunction = fieldFunction;
        this.view_name = view_name;
        this.scaleMin = scaleMin;
        this.scaleMax = scaleMax;
        this.stepSize = stepSize;
        this.startCoordinate = startCoordinate;
        this.endCoordinate = endCoordinate;
        this.focalpoint = focalpoint;
        this.pointOfView = pointOfView;
        this.zoom = zoom;
    }
}
