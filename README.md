# klee-service

## System Requirements
___
- Windows 7+ or higher
- MacOS 10.9+ or higher
- Python 3.12.4+
- pip 23.2.1+

## Installation
___

### 1.Clone the repository

```bash
git clone https://github.com/klee-contrib/klee-service.git
```
### 2.Install the dependencies

#### Windows

After entering the root directory of the project, use Win + R to open the Run dialog, type cmd, and press Enter.

Type the following command to create a virtual environment:
```bash
python -m venv venv
```
Source the virtual environment:
```bash
venv\Scripts\activate
```

Use the following command to install the package:
```bash
pip install -r requirements.txt
```

#### MasOS/Linux

After entering the root directory of the project, use the following command to create a virtual environment:
```bash
python3 -m venv venv
```
Source the virtual environment:
```bash
source venv/bin/activate
```

### 3.Add some code to the file
The file 'venv/Lib/site-packages/llama_index/llms/ollama/base.py' is missing some code.
You need to add the following code to the file, find method 'def _get_response_token_counts(self, raw_response: dict)' and add the following code:
```python
    def _get_response_token_counts(self, raw_response: dict) -> dict:
        """Get the token usage reported by the response."""
        if raw_response["done"] == False:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        # The above code is the code that needs to be added
        try:
            prompt_tokens = raw_response["prompt_eval_count"]
            completion_tokens = raw_response["eval_count"]
            total_tokens = prompt_tokens + completion_tokens
        except KeyError:
            return {}
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
```


Use the following command to install the package:
```bash
pip install -r requirements.txt
```

## Usage
___

Use the following command to run the server:
```bash
python app.py
```

The default environment is local, if you want to run the server in the background, you can use the following command:
```bash
python app.py --env youur_env_name
```

The default port number is 6190, if you want to change the port number, you can use the following command:
```bash
python app.py --port 6191
```

## Configuration
___

If you want to use the OpenAI API or Llama Cloud API, you need to provide your API key. 
Create a .env file in the root directory of the project and add the following content:

```env
OPENAI_KEY="your_openai_key"
LLAMA_CLOUD_API_KEY="your_llama_cloud_api_key"
```

## Build
___
### Windows
Use the following command to build the project:
```bash
pyinstaller --uac-admin --icon=klee-main.ico  --additional-hooks-dir=./hooks --add-data "./venv/Lib/site-packages/llama_index/core/agent/react/templates/system_header_template.md;./llama_index/core/agent/react/templates" --hidden-import=pydantic.deprecated.decorator --hidden-import=tiktoken_ext.openai_public --hidden-import=tiktoken_ext  -D main.py --clean    
```

### MacOs/Linux
Use the following command to build the project:
```bash 
pyinstaller --uac-admin --icon=klee-main.ico  --additional-hooks-dir=./hooks --add-data "./venv/Lib/site-packages/llama_index/core/agent/react/templates/system_header_template.md:./llama_index/core/agent/react/templates" --hidden-import=pydantic.deprecated.decorator --hidden-import=tiktoken_ext.openai_public --hidden-import=tiktoken_ext  -D main.py --clean    
```

## Technologies
___
- Python
- LlamaIndex
- Ollama
- Sqlalchemy
- FastAPI

