Backend for PetVac. Written in Python using Flask.

Please install these libraries first through `pip`:
```
pip install flask pandas scikit-learn ollama pydantic flask_cors
```

Add `pickle` to pip if needed.

Then, download Ollama from its website [here](https://ollama.com/download).
After installation, pull an LLM model through cmd (this project uses `llama3.2:3b-instruct-q6_K`) by:
```
ollama pull <model_name>
```
Run the server via:
```
python main.py
```
Note that this has no UI defined at `/`.

There are two routes:

`POST /llm`
- Returns a JSON output containing the pets and their allergies.
- Requires an input.

`POST /suggest`
- Returns the url, product_name, and the image_url of the suggested products.
- Needs num (number of suggestions) and the list of dogs similar in format from `/llm`.
