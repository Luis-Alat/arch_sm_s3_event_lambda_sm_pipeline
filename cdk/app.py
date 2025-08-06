import aws_cdk as cdk

from cdk.lambdas.lambda_stack import CdkLambdaStack
from cdk.api.api_stack import CdkApiStack
from cdk.pipeline.sagemaker_pipeline_stack import CdkSagemakerPipelinesStack

app = cdk.App()

lambda_stack = CdkLambdaStack(app, "LambdaStack")
api_stack = CdkApiStack(app, "ApiStack", lambda_function=lambda_stack.lambda_create_end)
sm_pipe_stack = CdkSagemakerPipelinesStack(app, "SagemakerPipelineStack")

app.synth()
