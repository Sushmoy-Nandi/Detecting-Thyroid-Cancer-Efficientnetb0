from src.ThyroidCancer.config.configuration import ConfigurationManager
from src.ThyroidCancer.components.federated_learning import FederatedLearning
from src.ThyroidCancer import logger


STAGE_NAME = "Federated Learning stage"


class FederatedLearningPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        federated_learning_config = config.get_federated_learning_config()
        federated_learning = FederatedLearning(config=federated_learning_config)
        federated_learning.run_federated_learning()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FederatedLearningPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
