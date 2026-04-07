try:
    from openenv_core.config import load_config
    config = load_config("openenv.yaml")
    print(f"Config loaded. Found {len(config.tasks)} tasks.")
    for task in config.tasks:
        print(f"Task: {task.id}")
except Exception as e:
    import traceback
    traceback.print_exc()
