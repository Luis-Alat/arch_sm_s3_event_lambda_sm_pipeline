import aws_cdk as cdk

from lambdas.lambda_stack import CdkLambdaStack
from api.api_stack import CdkApiStack
from pipeline.sagemaker_pipeline_stack import CdkSagemakerPipelinesStack
from event.event_bridge_stack import CdkEventStack

app = cdk.App()

lambda_stack = CdkLambdaStack(app, "LambdaStack")
api_stack = CdkApiStack(app, "ApiStack", lambda_function=lambda_stack.lambda_create_end)
sm_pipe_stack = CdkSagemakerPipelinesStack(app, "SagemakerPipelineStack")
event_rule = CdkEventStack(app, "EventBridgeRuleStack")

app.synth()
