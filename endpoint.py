from sagemaker.tensorflow import TensorFlowModel

model = TensorFlowModel(
    model_data="s3://mi-bucket/model.tar.gz",
    role=role,
    framework_version="2.10",
    entry_point="inference.py",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)