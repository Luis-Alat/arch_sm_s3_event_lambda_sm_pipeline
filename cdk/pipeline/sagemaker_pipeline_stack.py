from aws_cdk import Stack, aws_sagemaker
from constructs import Construct
import json


class CdkSagemakerPipelinesStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:

        super().__init__(scope, id, **kwargs)

        pipeline_name = "PipelineSkLernLoanDefault"
        pipeline_role_arn = "arn:aws:iam::007863746889:role/sagemakerS3"


        with open("pipeline/sagemaker/loan_default_pipeline.json", "r") as f:
            pipeline_definition = f.read()


        #pipeline_definition_property = aws_sagemaker.CfnPipeline.PipelineDefinitionProperty(
        #    pipeline_definition_body=pipeline_definition
        #)


        self.sm_pipeline = aws_sagemaker.CfnPipeline(
            self,
            pipeline_name,
            pipeline_name=pipeline_name,
            role_arn=pipeline_role_arn,
            pipeline_definition={"PipelineDefinitionBody": pipeline_definition}
        )