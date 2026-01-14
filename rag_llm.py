from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

vector_store_path = "vectorstores/faiss_store"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = FAISS.load_local(
    vector_store_path,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}      
)

from langchain_core.runnables import RunnableLambda

docs2str = RunnableLambda(
    lambda docs: "\n\n".join(d.page_content for d in docs)
)


class EcoTask(TypedDict):
    task_num : int
    task_description : str

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(EcoTask)

from langchain_core.prompts import ChatPromptTemplate
template = """Answer the question using the provided context:
{context}
Return task number and task content only.
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()} | prompt | structured_llm
)

res = rag_chain.invoke("What is question 2a about?")
print(res)  

#Handle follow up questions
history = [
    "Question 2a is a game theory problem about a simultaneous game between players A and B choosing North Sea or South Sea.",
    "The Nash equilibrium in Question 2a is both players choosing North Sea, earning $17,500 each.",
    "Question 2a resembles a prisoner's dilemma because both players could earn more by choosing South Sea."
]

#Create history aware retriever
rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the latest user question,
rewrite the question so that it is fully standalone and unambiguous.
Do NOT answer the question.

Conversation history:
{history}

Latest question:
{question}

Standalone search query:
""")
rewrite_prompt_chain = rewrite_prompt | llm | StrOutputParser()

from langchain_core.runnables import RunnableLambda
#x is input prompt
history_aware_retriever = (
    RunnableLambda(
        lambda x: {
            "history": "\n".join(x["history"]),
            "question": x["question"],
        }
    )
    | rewrite_prompt_chain
    | retriever
)

#rag chain with history aware retriever
history_aware_rag_chain = (
    {"context": history_aware_retriever | docs2str, "question": RunnableLambda(lambda x: x["question"])} 
    | prompt 
    | structured_llm
)

res2 = history_aware_rag_chain.invoke({
    "question": "next question content of that",
    "history": history
})
print(res2)


