from mldesigner import command_component, Input

@command_component(environment="environment.aml.yaml")
def convert_uri_file_to_int(uri_file: Input(type="uri_file")) -> int:
    with open(uri_file, 'r') as file:
        return int(file.read())
