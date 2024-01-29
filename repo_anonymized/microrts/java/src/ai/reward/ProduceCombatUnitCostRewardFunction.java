package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitType;
import util.Pair;

public class ProduceCombatUnitCostRewardFunction extends RewardFunctionInterface {
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs) {
        reward = 0.0;
        done = false;
        for (Pair<Unit, UnitAction> p : te.getActions()) {
            if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_PRODUCE
                    && p.m_b.getUnitType() != null) {
                UnitType produceUnitType = p.m_b.getUnitType();
                if (produceUnitType.name.equals("Light") || produceUnitType.name.equals("Heavy")
                        || produceUnitType.name.equals("Ranged")) {
                    reward += produceUnitType.cost;
                }
            }
        }
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }
}
