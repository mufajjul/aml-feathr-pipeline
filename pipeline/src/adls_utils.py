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
