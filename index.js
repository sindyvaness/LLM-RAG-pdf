import { ChatOllama } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import chalk from "chalk";
import promptSync from "prompt-sync";
import fs from "fs";


const OLLAMA_URL = "http://localhost:11434";

const chatModel = new ChatOllama({ 
    model: "mistral", 
    baseUrl: OLLAMA_URL, 
    temperature: 1, 
    verbose : false 
});

// let path = "./pdfs/Resume-SysEng-Sindy-Montano.pdf";

// ask user for the path
const prompt = promptSync();
let path = process.argv[2];//prompt(chalk.yellow("Enter the path of the document (e.g., ./pdfs/file.pdf): "));

// optional: trim spaces
path = path.trim();

// check if file exists
if (!fs.existsSync(path)) {
  console.log("File not found. Please check the path and try again.");
  process.exit(1);
}

// loading the pdf
const loader = new PDFLoader(path);
const docs = await loader.load();
// console.log("docs size", docs.length);

// Splits the PDF to smaller chunk
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500, // characters
    chunkOverlap: 100, // rule of thumbs: chunkOverlap ≈ 10% to 20% of chunkSize
  });

const allSplitsDocs = await textSplitter.splitDocuments(docs);

// console.log("allSplits size:", allSplitsDocs.length);

// Creating the vectors from the documents (embeddings)
const embeddingsModel = new OllamaEmbeddings({  model: "mxbai-embed-large" });


// Stores them in Vector Store
const vectorStore = await MemoryVectorStore.fromDocuments(allSplitsDocs, embeddingsModel);

const retriever = vectorStore.asRetriever({
  searchType: "mmr",
  searchKwargs: {
    fetchK: 3,
  },
});

// Creating the conversational memory
const memory = new BufferMemory({
  memoryKey: "chat_history",
  inputKey: "question",
  outputKey: "answer",
  returnMessages: true,
});

// user can ask questions about the PDF file
console.log(chalk.green(" The Chat RAG is ready. You can ask any question about the document: "));
console.log(chalk.green("Write '/bye'to end the chat.\n"));

let goOut = true;
while(goOut) {
    const q = prompt(chalk.yellow("Your question: "));

    if (q.trim().toLowerCase() === "/bye") {
        console.log(chalk.cyan("Good bye"));
        process.exit(0);
        goOut = false;
    }

    let rgaContext = await retriever.invoke(q);

    const promptTemplate = ChatPromptTemplate.fromMessages(
        ["system", `Use the following context to answer the question. 
        If you don't kno the answer just say 'I don't know'
        keep the answer short and concise please sound natural and friendly
        context: {context}, question: {question}`],
        new MessagesPlaceholder("chat_history"),
        ["human", "Context: {context}, question: {question}"],
        );

    const chain = promptTemplate.pipe(chatModel);

    // Using the context
    // console.log(chalk.yellow(q));
    let response = await chain.invoke({ context: rgaContext, question: q });
    console.log(chalk.green(response.content));

};








