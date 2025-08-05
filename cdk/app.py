import aws_cdk as cdk

from lambda_stack import CdkLambdaCiStack
from api_stack import ApiStack

app = cdk.App()

lambda_stack = CdkLambdaCiStack(app, "LambdaStack")
api_stack = ApiStack(app, "ApiStack", lambda_function=lambda_stack.lambda_create_end)

app.synth()
