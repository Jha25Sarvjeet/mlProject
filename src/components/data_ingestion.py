import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestioinConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestioinConfig()

    def initiate_data_ingestion(self):
        # print("Entered the data ingestion method or component")
        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv('notebook\data\stud.csv')
            # print("Read the dataset as dataframe")
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('ingestion of the data is completed')
            

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path,
                   )


        except Exception as e:
            # print("error occured ",str(e))
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_array=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    model_r2_score=model_trainer.initiate_model_trainer(train_array=train_array,test_array=test_array)
    print(model_r2_score
          )
    



