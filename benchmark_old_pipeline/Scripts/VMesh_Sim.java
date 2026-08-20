// Simcenter STAR-CCM+ macro: run.java
// Written by Simcenter STAR-CCM+ 18.04.008
package macro;

import java.util.*;
import star.common.*;
import star.meshing.*;

public class VMesh_Sim extends StarMacro {
    @Override
    public void execute() {
        Simulation simulation = getActiveSimulation();
        if (simulation != null) {
            simulation.println("Generating volume mesh...");
            simulation.get(MeshPipelineController.class).generateVolumeMesh();
            simulation.println("Mesh generation complete.");
        } else {
            System.out.println("Error: No active simulation found.");
        }

        simulation.getSimulationIterator().run();

	simulation.saveState(resolvePath(simulation.getSessionPath()));
    }
}
