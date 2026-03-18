from src.ThyroidCancer.config.configuration import ConfigurationManager
from src.ThyroidCancer.components.explainability import Explainability
from src.ThyroidCancer import logger


STAGE_NAME = "Explainability stage"


class ExplainabilityPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        explainability_config = config.get_explainability_config()
        explainability = Explainability(config=explainability_config)
        explainability.load_model()
        explainability.generate_explanations(num_samples=5)



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ExplainabilityPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
