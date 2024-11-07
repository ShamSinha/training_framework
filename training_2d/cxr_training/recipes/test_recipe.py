from cxr_training.inference.inference import (
    run_inference_for_model_list,
)
from cxr_training.metrics.test_metric import Metrics
import glob


class TestRecipe:
    def __init__(self, args):
        self.args = args
        self.checkpoint_dir = f"{args.trainer.checkpoint_dir}/{args.trainer.description}/{args.trainer.model_file}"

        self.model_list = self.args.trainer.model_list
        self.is_inference = self.args.params.inference_run
        self.is_metric = self.args.params.metric_run
        self._set_model_list()
        self._set_tasks()

    def _set_model_list(self):
        """private setter method for model list"""

        if len(self.model_list) == 0:
            self.model_list = glob.glob(f"{self.checkpoint_dir}/*.ckpt", recursive=True)

        else:
            self.model_list = [self.checkpoint_dir + "/" + x for x in self.model_list]

        print(self.model_list)

    def _inference_task(self):
        """private method to inititalise models and dataloaders"""

        print("calculating inference")
        print("============self.args==============", self.args)
        inference_func = run_inference_for_model_list
        kwargs = {"args": self.args, "model_list": self.model_list}
        inf = inference_func(**kwargs)
        inf.run_inference()

    def _metrics_task(self):
        """private method to check various metrics such as auroc,accuracy from the test results"""

        print("calculating metrics")
        mt = Metrics(self.model_list, self.args)
        mt.run_metrics()

    def _set_tasks(self):
        self.tasks = []
        if self.is_inference:
            self.tasks.append(self._inference_task)
        if self.is_metric:
            self.tasks.append(self._metrics_task)

    def run(self):
        for task in self.tasks:
            task()
