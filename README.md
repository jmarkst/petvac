Backend for PetVac. Written in Python using Flask.

Please install these libraries first through `pip`:
```
pip install flask pandas scikit-learn ollama pydantic
```

Add `pickle` to pip if needed.

Then, download Ollama from its website [https://ollama.com/download here].
After installation, pull an LLM model through cmd (this project uses `llama3.2:3b-instruct-q6_K`) by:
```
ollama pull <model_name>
```
