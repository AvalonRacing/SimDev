// Simcenter STAR-CCM+ macro: run.java
// Written by Simcenter STAR-CCM+ 18.04.008
package macro;

import java.util.*;

import star.common.*;
import star.meshing.*;

public class SMesh extends StarMacro {
    @Override
    public void execute() {
        Simulation simulation = getActiveSimulation();
        if (simulation != null) {
            simulation.println("Generating surfacemesh...");
            simulation.get(MeshPipelineController.class).generateSurfaceMesh();
            simulation.println("Surfacemesh generation complete.");
        } else {
            System.out.println("Error: No active simulation found.");
        }
	simulation.saveState(resolvePath(simulation.getSessionPath()));
    }
}
