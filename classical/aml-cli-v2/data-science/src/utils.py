import yaml
import os
import logging
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from feathr import FeathrClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# container to hold common configs used throughout lifecycle (prep, train, deploy....)
# current limitation to push data directly to the storage account, hence user is instructed to store data (in readme) in this specific container
fs_config = {"data_container_name": "nyctaxi"}


def get_active_branch_name():
    """Get the name of the active branch"""
    head_dir = Path(os.path.join(
        Path(__file__).parent.parent.parent.parent.parent, ".git", "HEAD"))
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def get_yaml_file_path():
    """Get the path to the config.yaml file"""
    if get_active_branch_name() == "main":
        # 'main' branch: PRD environment
        logging.info("PRD environment, using config-infra-prod.yml")
        config_file = "config-infra-prod.yml"
    else:
        # 'develop' or feature branches: DEV environment
        logging.info("DEV environment, using config-infra-dev.yml")
        config_file = "config-infra-dev.yml"
    return os.path.join(Path(__file__).parent.parent.parent.parent.parent, config_file)


def get_credential():
    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True)
    return credential


def set_required_feathr_config(credential: DefaultAzureCredential):
    """Get configuration from config yaml file"""
    config_file_path = get_yaml_file_path()
    with open(config_file_path, "r") as f:
        logging.info("Reading  {} file".format(config_file_path))
        config = yaml.safe_load(f)

    # adding "fs" to namespace as this is waht we do in the infrastructure code to separate featurestore resorces
    resource_prefix = config.get("variables").get("namespace") + "fs"
    resource_postfix = config.get("variables").get("postfix")
    resource_env = config.get("variables").get("environment")
    logging.info("using resource prefix: {}, resource postfix: {} and environment: {},".format(
        resource_prefix, resource_postfix, resource_env))

    # Get all the required credentials from Azure Key Vault
    key_vault_name = "kv-"+resource_prefix+"-"+resource_postfix+resource_env
    synapse_workspace_url = "sy"+resource_prefix+"-"+resource_postfix+resource_env
    adls_account = "st"+resource_prefix+resource_postfix+resource_env
    adls_fs_name = "dl"+resource_prefix+resource_postfix+resource_env
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    secretName = "FEATHR-ONLINE-STORE-CONN"
    retrieved_secret = str(client.get_secret(secretName).value)

    # Get redis credentials; This is to parse Redis connection string.
    redis_port = retrieved_secret.split(',')[0].split(":")[1]
    redis_host = retrieved_secret.split(',')[0].split(":")[0]
    redis_password = retrieved_secret.split(',')[1].split("password=", 1)[1]
    redis_ssl = retrieved_secret.split(',')[2].split("ssl=", 1)[1]

    # Set appropriate environment variables for overriding feathr config
    os.environ['spark_config__azure_synapse__dev_url'] = f'https://{synapse_workspace_url}.dev.azuresynapse.net'
    os.environ['spark_config__azure_synapse__pool_name'] = 'spdev'
    os.environ['spark_config__azure_synapse__workspace_dir'] = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_project'
    os.environ['online_store__redis__host'] = redis_host
    os.environ['online_store__redis__port'] = redis_port
    os.environ['online_store__redis__ssl_enabled'] = redis_ssl
    os.environ['REDIS_PASSWORD'] = redis_password
    os.environ['FEATURE_REGISTRY__API_ENDPOINT'] = f'https://app{resource_prefix+resource_postfix+resource_env}.azurewebsites.net/api/v1'

    # Set common configs used throughout lifecycle (prep, train, deploy....)
    fs_config['feathr_output_path'] = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_output'
    fs_config['adls_account'] = adls_account


def get_feathr_client():
    credential = get_credential()
    set_required_feathr_config(credential=credential)
    config_file_path = os.path.join(
        Path(__file__).parent, "feathr_config.yaml")
    logging.info("config path: {}".format(config_file_path))
    return FeathrClient(config_path=config_file_path, credential=credential)

# The three methods below are used for writing/reading data from ADLS storage
def create_or_retrieve_file_system(service_client, file_system):
    '''Creates a new file system on a given service client, or returns an existing one.'''
    file_system_client = service_client.get_file_system_client(file_system)
    if file_system_client.exists():
        print("File system already exists:", file_system)
    else:
        file_system_client.create_file_system()
        print("File system created:", file_system)
    return file_system_client

def create_or_retrieve_directory(file_system_client, directory):
    directory_client = file_system_client.get_directory_client(directory)
    if directory_client.exists():
        print("Directory already exists:", directory)
    else:
        directory_client.create_directory()
        print("Directory created:", directory)
    return directory_client

def create_or_retrieve_file(directory_client, file):
    file_client = directory_client.get_file_client(file)
    if file_client.exists():
        print("File already exists:", file)
    else:
        file_client.create_file()
    return file_client
