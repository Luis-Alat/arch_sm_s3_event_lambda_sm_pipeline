from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import Stack

import json
from constructs import Construct

class CdkEventStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        bucket = s3.Bucket.from_bucket_name(
            self,
            "LoanDefaultBucket",
            "pipeline-test-ml-sklearn-randomforest-artifacts"
        )


        role = iam.Role.from_role_arn(
            self, 
            "EventBridgeRoleTargetPipeline",
            role_arn="arn:aws:iam::007863746889:role/service-role/EventBridge_Invoke_SageMaker_Pipeline"
        )

        events.CfnRule(
            self,
            "S3ToSageMakerPipelineEventBridgeRule",
            event_pattern={
                "source": ["aws.s3"],
                "detail-type": ["Object Created"],
                "detail": {
                    "bucket": {
                        "name": [bucket.bucket_name]
                    },
                    "object": {
                        "key": [{"prefix": "develop/data/raw/"}]
                    }
                }
            },
            targets=[
                events.CfnRule.TargetProperty(
                    id="SageMakerPipelineTarget",
                    arn="arn:aws:sagemaker:us-east-1:007863746889:pipeline/PipelineSkLernLoanDefault",
                    role_arn=role.role_arn,
                    input=json.dumps({
                        "PipelineName": "PipelineSkLernLoanDefault"
                    })
                )
            ]
        )
