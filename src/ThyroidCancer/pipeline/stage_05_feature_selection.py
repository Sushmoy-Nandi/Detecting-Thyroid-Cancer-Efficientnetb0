from src.ThyroidCancer.config.configuration import ConfigurationManager
from src.ThyroidCancer.components.feature_selection import FeatureSelection
from src.ThyroidCancer.components.data_ingestion import DataIngestion
from src.ThyroidCancer import logger


STAGE_NAME = "Feature Selection stage"


class FeatureSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        # Ensure split exists before feature selection
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.create_train_val_test_split()
        feature_selection_config = config.get_feature_selection_config()
        feature_selection = FeatureSelection(config=feature_selection_config)
        feature_selection.select_features()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureSelectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
