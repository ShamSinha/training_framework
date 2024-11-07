import hydra
from omegaconf import OmegaConf
from clearml import Task
from cxr_training.nnmodule.models.utils import get_class_from_str
from configs.pydantic_validation import FullConfig
import warnings

warnings.filterwarnings("ignore")


def main(args):
    # print(OmegaConf.to_yaml(args))

    if args.use_clearml:
        continue_id = getattr(args, "continue_id", None)
        if continue_id is not None:
            task = Task.init(
                project_name=args.trainer.project,
                task_name=args.trainer.model_file,
                tags=args.trainer.description,
                # reuse_last_task_id=True,
                continue_last_task=continue_id,
            )
        else:
            task = Task.init(
                project_name=args.trainer.project,
                task_name=args.trainer.model_file,
                tags=args.trainer.description,
                reuse_last_task_id=True,
                continue_last_task=True,
            )

        task.connect_configuration(vars(args), name="args_config")

    if args.trainer.recipe != "testing" and args.validate_config:
        "running args validation"
        FullConfig.model_validate(args)

    recipe_module = get_class_from_str(args.trainer.recipe_module)
    recipe = recipe_module(args)
    recipe.run()


if __name__ == "__main__":
    main_wrapper = hydra.main(version_base="1.3")
    main_wrapper(main)()
