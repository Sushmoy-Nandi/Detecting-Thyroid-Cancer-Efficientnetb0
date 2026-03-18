from src.ThyroidCancer.config.configuration import ConfigurationManager
from src.ThyroidCancer.components.evaluation import Evaluation
from src.ThyroidCancer import logger




STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
