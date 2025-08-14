from aws_cdk import Stack, aws_apigateway, CfnOutput
from constructs import Construct

class CdkApiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, *, lambda_function, **kwargs):
        
        super().__init__(scope, construct_id, **kwargs)

        api = aws_apigateway.RestApi(
            self,
            "RestApiGetPrediction",
            rest_api_name="GetPrediction",
        )

        predict_resource = api.root.add_resource("predict")
        predict_resource.add_method(
            "POST",
            aws_apigateway.LambdaIntegration(lambda_function)
        )

        CfnOutput(
            self,
            "ApiGetPredictionUrl",
            value=api.url,
        )

        CfnOutput(
            self,
            "ApiGetPredictionResourcePredict",
            value=predict_resource.resource_name
        )