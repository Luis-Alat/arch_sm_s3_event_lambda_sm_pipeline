from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_iam as iam
)
from constructs import Construct


class CdkLambdaStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):

        super().__init__(scope, construct_id, **kwargs)

        invocation_lambda_role = iam.Role.from_role_arn(
            self, 
            "LambdaRoleCallEndpoint",
            role_arn="arn:aws:iam::007863746889:role/CallEndpoint"
        )

        create_endpoint_lambda_role = iam.Role.from_role_arn(
            self,
            "LambdaRoleSeverlessDeploySagemakerPipeline",
            role_arn="arn:aws:iam::007863746889:role/ModelArtefacts"
        )


        lambda_call_end = _lambda.Function(
            self,
            "LambdaFunctCallEndpoint",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="CallEndpoint.lambda_handler",
            code=_lambda.Code.from_asset("lambdas/call_endpoint"),
            role=invocation_lambda_role,
            function_name="CallEndpoint"
        )

        lambda_create_end = _lambda.Function(
            self,
            "LambdaFunctCreateEndpoint",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="SeverlessDeploySagemakerPipeline.lambda_handler",
            code=_lambda.Code.from_asset("lambdas/severless_deploy_endpoint"),
            role=create_endpoint_lambda_role,
            function_name="SeverlessDeploySagemakerPipeline"
        )


