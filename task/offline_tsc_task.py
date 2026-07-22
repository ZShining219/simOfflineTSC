"""Task registration for the separate pure-offline entry point."""

from common.registry import Registry
from task.task import BaseTask


@Registry.register_task('offline_tsc')
class OfflineTSCTask(BaseTask):
    def run(self):
        self.trainer.train()
