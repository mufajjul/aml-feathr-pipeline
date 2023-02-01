import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", type=str, help="Azure subscription id.")
    parser.add_argument("--resource_group", type=str, help="Name of the resource group.")
    parser.add_argument("--workspace_name", type=str, help="Name of the AML workspace.")
    parser.add_argument("--config_path", type=str, help="Path where to save the AML config file.")
    args = parser.parse_args()

    aml_config = f"""
    {{
    "subscription_id": "{args.subscription_id}",
    "resource_group": "{args.resource_group}",
    "workspace_name":"{args.workspace_name}"
    }}
    """

    with open(args.config_path, "w") as file:
        file.write(aml_config)

    print("AML config file created successfully at:", args.config_path)




if __name__ == '__main__':
    main()