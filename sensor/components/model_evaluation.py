from sensor.predictor import ModelResolver
from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
import pandas as pd
import os, sys
from sensor.logger import logging
from sensor.utils import load_object
from sensor.config import TARGET_COLUMN
from sklearn.metrics import f1_score


class ModelEvaluation:
    
    def __init__(self, 
        model_eval_config:config_entity.ModelEvaluationConfig, 
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact
        ):
        try:
            logging.info(f"{'>>'*20} Model evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self, ) -> artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model, we will compare which model is best trained and 
            logging.info(f"if saved model folder has model, we will compare which model is best trained from save model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                impoved_accuracy=None)
                logging.info(f"Model evaluation artifact :{model_eval_artifact}")
                return model_eval_artifact

            #finding location of transformer model and target encoder
            logging.info("finding location of transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            #previously trained model object
            logging.info("Previous trained objects of transformer, model and target encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            #currently trained model objects
            logging.info("currently trained model objects")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)


            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df =  test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)
            # accuracy using previously trained model
    
            input_arr = transformer.transform(test_df)
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")

            #accuracy using current trained model
            logging.info("accuracy using current trained model")
            input_arr = transformer.transform(test_df)
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)
            print(f"Prediction using trained model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")

            if current_model_score < previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("current trained model is not better than previously trained model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, impoved_accuracy=current_model_score - previous_model_score)
            logging.info(f"model_eval_artifact: {model_eval_artifact}")               
            return model_eval_artifact

            
        except Exception as e:
            raise SensorException(e, sys)

