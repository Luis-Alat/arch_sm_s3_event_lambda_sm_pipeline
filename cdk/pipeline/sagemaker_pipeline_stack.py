from aws_cdk import Stack, aws_sagemaker
from constructs import Construct
import json


class CdkSagemakerPipelinesStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:

        super().__init__(scope, id, **kwargs)

        # 1. Load the JSON file into a Python dictionary.
        with open("pipeline/sagemaker/loan_default_pipeline.json", "r") as f:
            pipeline_definition_dict = json.load(f)

        pipeline_name = "PipelineSkLernLoanDefault"
        pipeline_role_arn = "arn:aws:iam::***:role/sagemakerS3"

        # 2. Create a PipelineDefinitionProperty object.
        # This object expects a dictionary as the pipeline_definition_body.
        # It's a bit counter-intuitive, but this is the structure that the CDK needs.
        pipeline_definition_property = aws_sagemaker.CfnPipeline.PipelineDefinitionProperty(
            # Pass the Python dictionary directly
            pipeline_definition_body=pipeline_definition_dict
        )

        # 3. Create the CfnPipeline resource.
        # Pass the PipelineDefinitionProperty object to the pipeline_definition property.
        self.sm_pipeline = aws_sagemaker.CfnPipeline(
            self,
            pipeline_name,
            pipeline_name=pipeline_name,
            role_arn=pipeline_role_arn,
            pipeline_definition=pipeline_definition_property,
        )