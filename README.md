## 📁 Project Structure

![alt text](other/img/image.png)

```bash

.
├── cdk
│   ├── api
│   ├── event
│   ├── lambdas
│   │   ├── call_endpoint
│   │   └── severless_deploy_endpoint
│   ├── pipeline
│   │   └── sagemaker
│   └── roles
│       ├── CallEndpoint
│       │   ├── attachedPolicies
│       │   └── inlinePolicies
│       ├── EventBridge_Invoke_SageMaker_Pipeline
│       │   ├── attachedPolicies
│       │   └── inlinePolicies
│       ├── ModelArtefacts
│       │   ├── attachedPolicies
│       │   └── inlinePolicies
│       ├── sagemakerS3
│       │   ├── attachedPolicies
│       │   └── inlinePolicies
│       └── TriggerS3UpdateTrainData
│           ├── attachedPolicies
│           └── inlinePolicies
├── experiments
│   ├── data
│   │   ├── processed
│   │   └── raw
│   ├── models
│   └── notebooks
│       └── tmp
├── other
│   └── img
├── pipeline
│   └── steps
└── test
    ├── integration
    └── unit

```
