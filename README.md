This app use Ollama framework to do the following:

Implement a RAG system where it can take the path of a PDF file from user then
•Splits the PDF to smaller chunk
•Stores them in Vector Store
•When ready, notifies user to ask questions about the PDF file
•Repeat the Q & A until the user types /bye
•Then exit

Requirements:
-------------
1. Ollama must be installed in the system and working in: http://localhost:11434
2. The Ollama must have the following LLM models: "mistral" and "mxbai-embed-large"

Some required packages:
Run the following command:
npm install 

Execute the app:
----------------
To execute the app you must provide the path of the pdf file.

=> node index.js file.pdf

