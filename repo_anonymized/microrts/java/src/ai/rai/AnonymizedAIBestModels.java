package ai.rai;

import ai.core.AI;
import rts.units.UnitTypeTable;

public class AnonymizedAIBestModels extends AnonymizedAI {
    public AnonymizedAIBestModels(UnitTypeTable a_utt) {
        super(100, -1, a_utt, 0, 1, true);
    }

    public AnonymizedAIBestModels(int mt, int mi, UnitTypeTable a_utt) {
        super(mt, mi, a_utt, 0, 1, true);
    }

    public AnonymizedAIBestModels(int mt, int mi, UnitTypeTable a_utt, int overrideTorchThreads,
            int pythonVerboseLevel) {
        super(mt, mi, a_utt, overrideTorchThreads, pythonVerboseLevel, true);
    }

    public AI clone() {
        if (DEBUG >= 1)
            System.out.println("AnonymizedAIBestModels: cloning");
        return new AnonymizedAIBestModels(TIME_BUDGET, ITERATIONS_BUDGET, utt, OVERRIDE_TORCH_THREADS,
                PYTHON_VERBOSE_LEVEL);
    }
}
