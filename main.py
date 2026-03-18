from src.ThyroidCancer import logger
from src.ThyroidCancer.pipeline.stage_01_ingestion import DataIngestionTrainingPipeline
from src.ThyroidCancer.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.ThyroidCancer.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from src.ThyroidCancer.pipeline.stage_04_evaluation import EvaluationPipeline
from src.ThyroidCancer.pipeline.stage_05_feature_selection import FeatureSelectionPipeline
from src.ThyroidCancer.pipeline.stage_06_federated_learning import FederatedLearningPipeline
from src.ThyroidCancer.pipeline.stage_07_explainability import ExplainabilityPipeline
from dotenv import load_dotenv

load_dotenv()


# # Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 3: Feature Selection
STAGE_NAME = "Feature Selection stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   feature_selection = FeatureSelectionPipeline()
   feature_selection.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 4: Federated Learning (alternative to centralized training)
STAGE_NAME = "Federated Learning stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   federated_learning = FederatedLearningPipeline()
   federated_learning.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Optional: Stage 3b - Centralized Training (if not using federated learning)
# STAGE_NAME = "Training"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    model_trainer = ModelTrainingPipeline()
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# Stage 5: Evaluation
STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 6: Explainability
STAGE_NAME = "Explainability stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   explainability = ExplainabilityPipeline()
   explainability.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
