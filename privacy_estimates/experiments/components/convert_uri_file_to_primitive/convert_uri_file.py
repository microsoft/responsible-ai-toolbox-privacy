from mldesigner import command_component, Input

# TODO: This could be also solved by assigning a dict with keys conda_file and image to the environment parameter
# This should avoid many warnings about componentenvironment name not being able to set

@command_component(environment="environment.aml.yaml")
def convert_uri_file_to_int(uri_file: Input(type="uri_file")) -> int:
    with open(uri_file, 'r') as file:
        return int(file.read())
