from azure.ai.ml.entities import Environment

dependencies_dir = "./"
custom_env_name = "feathr-env"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Feathr environment",
    tags={"python": "3.9.12"},
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)