import express from "express";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { Annotation, MemorySaver, MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
dotenv.config();

const app = express();
const port = 5000;
app.use(express.json());

// without langchain

// const ai = new GoogleGenAI({
//   apiKey: process.env.GEMINI_API_KEY,
// });

// app.post("/ai", async (req, res) => {
//   const { input } = req.body;
//   const response = await ai.models.generateContent({
//     model: "gemini-3.8-flash",
//     contents: [
//       {
//         role: "system",
//         parts: [{ text: "you are a assistant and your name is jarvis." }],
//       },
//       {
//         role: "user",
//         parts: [{ text: input }],
//       }
//     ]
//   })
//   return res.status(200).json({ "ai:": response.text });
// })

// with langchain

const tool = new TavilySearch({
  maxResults: 5,
  topic: "general",
});

const checkPointer=new MemorySaver()

const tools=[tool]
const toolNode=new ToolNode(tools)

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxTokens: 1024,
  maxRetries: 2,
}).bindTools(tools);

const callLLM=async (state)=>{
  console.log("state:",state);
  
  const response = await llm.invoke([
    {
      role: "system",
      content: `
      You are Jarvis, a helpful AI assistant.

      Rules:
      1. Answer general questions using your knowledge.
      2. For questions about current weather, news, live information,
        or information requiring internet access, use tavily_search.
      3. Provide a clear and accurate answer using the search results.
      4. Never invent search results.
      5. If the tool is not needed, answer directly.
      `
    },
      ...state.messages
  ]);

  return {messages:[response]}
}

const shouldContinue = (state) => {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage.tool_calls?.length) {
    return "tools";
  }

  return "__end__";
};


const graph=new StateGraph(MessagesAnnotation)
.addNode("agent",callLLM)
.addNode("tools",toolNode)
.addEdge("__start__","agent")
.addEdge("tools","agent")
.addConditionalEdges("agent",shouldContinue)
.compile({checkpointer:checkPointer})


app.post("/ai", async (req, res) => {
  const { input } = req.body;

  const response=await graph.invoke(
    {messages:[
    {role:"human",
      content:input
    }
  ]},
  {configurable:{thread_id:"user123"}}
)
  console.log(response.messages);
  
  return res.status(200).json({ "ai:": response.messages[response.messages.length-1].content });
});

app.get("/", (req, res) => {
  return res.json({ message: "hello from Level4" });
});

app.listen(port, () => {
  console.log("server started");
});
