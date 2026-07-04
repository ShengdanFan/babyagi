from functionz.core.framework import func
import json
import litellm
import os

@func.register_function(
    metadata={
        "description": "A chat application that interacts with LiteLLM and executes selected functions from the database."
    },
    imports=["litellm", "json"],
    dependencies=["get_function_wrapper", "execute_function_wrapper"],
    key_dependencies=["openai_api_key"]
)
def chat_with_functions(chat_history, available_function_names) -> str:
    def map_python_type_to_json(python_type: str) -> dict:
        type_mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {"type": "string"}},
            "dict": {"type": "object"},
            "Any": {"type": "string"}
        }
        return type_mapping.get(python_type, {"type": "string"})

    litellm.set_verbose = True
    os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')

    chat_context = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    if not isinstance(chat_history, list):
        raise ValueError("chat_history must be a list of messages.")

    for message in chat_history:
        if not isinstance(message, dict):
            raise ValueError("Each message in chat_history must be a dictionary.")
        role = message.get('role')
        content = message.get('message')
        if role not in ['user', 'assistant', 'system']:
            raise ValueError("Message role must be 'user', 'assistant', or 'system'.")
        if not isinstance(content, str):
            raise ValueError("Message content must be a string.")
        chat_context.append({"role": role, "content": content})

    if isinstance(available_function_names, str):
        available_function_names = [name.strip() for name in available_function_names.split(',') if name.strip()]
    elif isinstance(available_function_names, list):
        available_function_names = [name.strip() for name in available_function_names if isinstance(name, str) and name.strip()]
    else:
        raise ValueError("available_function_names must be a string or a list of strings.")

    if not available_function_names:
        raise ValueError("No valid function names provided in available_function_names.")

    tools = []
    for func_name in available_function_names:
        function_data = get_function_wrapper(func_name)
        if function_data:
            tool = {
                "type": "function",
                "function": {
                    "name": function_data['name'],
                    "description": function_data['metadata']['description'],
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }
            for param in function_data.get('input_parameters', []):
                json_schema = map_python_type_to_json(param['type'])
                tool['function']['parameters']['properties'][param['name']] = {
                    **json_schema,
                    "description": param.get('description', '')
                }
                if param.get('required', False):
                    tool['function']['parameters']['required'].append(param['name'])
            tools.append(tool)
        else:
            raise ValueError(f"Function '{func_name}' not found in the database.")

    response = litellm.completion(
        model="deepseek/deepseek-v4-flash",
        messages=chat_context,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response['choices'][0]['message']
    tool_calls = response_message.get('tool_calls', [])

    if tool_calls:
        chat_context.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            tool_call_id = tool_call['id']
            try:
                function_response = execute_function_wrapper(function_name, **function_args)
            except Exception as e:
                function_response = f"Error executing function '{function_name}': {str(e)}"
            if not isinstance(function_response, str):
                function_response = json.dumps(function_response)
            chat_context.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })
        second_response = litellm.completion(
            model="deepseek/deepseek-v4-flash",
            messages=chat_context
        )
        assistant_response = second_response['choices'][0]['message']['content']
        return assistant_response
    else:
        assistant_response = response_message.get('content', '')
        return assistant_response
