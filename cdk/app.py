import aws_cdk as cdk

from lambda_stack import CdkLambdaStack
from api_stack import CdkApiStack

app = cdk.App()

lambda_stack = CdkLambdaStack(app, "LambdaStack")
api_stack = CdkApiStack(app, "ApiStack", lambda_function=lambda_stack.lambda_create_end)

app.synth()
