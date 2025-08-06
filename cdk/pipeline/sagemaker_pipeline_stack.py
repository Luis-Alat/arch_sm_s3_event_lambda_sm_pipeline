from aws_cdk import Stack, aws_sagemaker
from constructs import Construct
import json


class CdkSagemakerPipelinesStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:

        super().__init__(scope, id, **kwargs)

        # 1. Load the JSON file into a Python dictionary.
        with open("pipeline/sagemaker/loan_default_pipeline.json", "r") as f:
            pipeline_definition_dict = json.load(f)

        # 2. Convert the Python dictionary to a JSON string.
        # This is the crucial step to satisfy the 'str' type requirement.
        pipeline_definition_json_string = json.dumps(pipeline_definition_dict)

        pipeline_name = "PipelineSkLernLoanDefault"
        pipeline_role_arn = "arn:aws:iam::***:role/sagemakerS3"

        # 3. Create a CfnPipeline.PipelineDefinitionProperty object.
        # This object's 'pipeline_definition_body' property explicitly expects a string.
        pipeline_definition_property = aws_sagemaker.CfnPipeline.PipelineDefinitionProperty(
            pipeline_definition_body=pipeline_definition_json_string
        )

        # 4. Create the CfnPipeline resource and pass the property object.
        self.sm_pipeline = aws_sagemaker.CfnPipeline(
            self,
            pipeline_name,
            pipeline_name=pipeline_name,
            role_arn=pipeline_role_arn,
            pipeline_definition=pipeline_definition_property,
        )