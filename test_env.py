# pyre-ignore-all-errors
from env import FinOpsEnv
# pyre-ignore[21]
from schemas.models import Action, ActionType

def main():
    env = FinOpsEnv(task_id="sandbox_cleanup")
    obs, info = env.reset()
    print(f"Initial obs: step={obs.step_number}, spend={obs.total_monthly_spend}")
    
    # Take an action
    action = Action(
        action_type=ActionType.ANALYZE,
        target_resource=obs.resources[0].resource_id,
        parameters={}
    )
    obs, reward, done, info = env.step(action)
    print(f"After action: reward={reward}, done={done}, msg={obs.system_message}")
    
    # Finalize
    action = Action(action_type=ActionType.FINALIZE_PLAN, parameters={})
    obs, reward, done, info = env.step(action)
    print(f"Final state: total_savings={info['total_savings']}, grade_score={reward}")

if __name__ == "__main__":
    main()
