import express from "express";
import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
dotenv.config();

const app = express();
const port = 5000;
app.use(express.json());

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxTokens: 1024,
  maxRetries: 2,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "grocery-store",
});

const upload = async () => {
  const pdfPath = "./knowledge.pdf";
  const buffer = fs.readFileSync(pdfPath);
  const pdfResult = new PDFParse({ data: buffer });
  const result = await pdfResult.getText();
  const text = result.text;
  const spilitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await spilitter.createDocuments([text]);
  await vectorStore.addDocuments(docs);
};

app.post("/ai", async (req, res) => {
  const { input } = req.body;

  const docs = await vectorStore.similaritySearch(input, 5);
  const context = docs.map((d) => d.pageContent).join("/n");

  const response = await llm.invoke([
    new SystemMessage(`you are a RAG AI Assistant.
        
      STRICT RULES:
      - Answer only from context
      - Do not use outside knowledge
      - if answer not found say:
        "I dont't know from upload PDF."
        
      context:
      ${context}`),
    new HumanMessage(input)  
  ]);

  return res.status(200).json({ai:response.content});
});

app.get("/", (req, res) => {
  return res.json({ message: "hello from Level4" });
});

app.listen(port, () => {
  console.log("server started");
});
