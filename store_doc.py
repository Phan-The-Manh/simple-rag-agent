from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

vector_store_path = "vectorstores/faiss_store"

def create_db_from_text():
    raw_text = """HE5091 Semester 1 Examination 2023-2024 Marker’s Report

The performance of students in this examination is satisfactory.  Majority of students are able to secure a pass and a large percentage of students is able to obtain a good grade.  However, some students did not complete all the questions within the 2½ hour duration.  Time management is important to ensure sufficient time to attempt all the 4 questions.

Question 1

1(a) is a question on market equilibrium analysis.  With demand for seafood and supply of seafood both decreases concurrently, the effect is quantity decreases but change in price is uncertain.  There 3 possible outcomes when both demand and supply curve shifts left, with quantity decreases but price can be higher, lower or unchanged.  Many students are able to provide the 3 diagrams needed but some students only draw one diagram with one analysis and hence the answer is incomplete.

1(b) is a question on rational spending rule where MUX/PX = MUY/PY.  The statement is uncertain because even if MUX > MUY, consumer will only buy more X and less Y if PX is less than or equal to PY.  Many students mistakenly conclude the statement is true which is incorrect.

1(c) is a quantitative question requires solving of simultaneous equations and computation of welfare.  Most students are able to equate with demand and supply to solve for the equilibrium price and quantity under perfect competition and the required consumer surplus and producer surplus.  Under monopoly, the optimal output is obtained by MR = MC and the optimal price is obtained by substituting this output into the demand equation.  Some students did not obtain the correct equation of MR, which is P = 250 – 0.5Q to equate with MC which is the supply equation.  This will affect their computation of the optimal price and quantity, and hence the required consumer surplus, producer surplus and deadweight loss under monopoly.

Question 2

2(a) is a game theory question which is well attempted by the students.  Part (i) is a simultaneous game and most students are able to obtain the payoff matrix and solve for the Nash equilibrium which is both A and B selects North Sea and each earn $17,500.  Most students are also correctly pointed out this game is a prisoner’s dilemma game since both A and B can earn more if they selects South Sea.  Part (ii) is a sequential game where most students are able to draw the decision tree diagram correctly with A as the first mover and solve the game correctly using roll-back method with A selects North Sea in Stage 1 and B selects North Sea in Stage 2 and both earn $17,500.    

2(b) is an externality question using Coase Theorem and the performance of students in this question is mixed.  In Part (i), Bob will not install soundproofing and this outcome is socially efficient since the joint payoff without soundproofing is higher than with soundproofing.  Bob has the legal right to sing without soundproofing and it is not worth for Shawn to compensate him.  In Part (ii), Bob will still not install soundproofing and this outcome is socially efficient.  He will pay $80 to compensate Shawn for the noise he made.  Not many students conclude that attaining the efficient outcome does not depend on who has the legal right.

2(c) is a question on public good.  Many students did not provide the correct answer which is the garden will not be approved.  The tax is $12 per head but only A will approve it.  B and C will not approve since their marginal benefits are less than the tax they have to pay.  This outcome is not socially efficient since the total benefit with the garden exceeds the total cost.

Question 3

3(a) is a question of GDP and CPI computation and students did well in this question.  Most students are able to compute nominal GDP which is current year price x current year quantity and the real GDP which is base year price x current year quantity.  They are also able to compute the CPI which is current year cost of base year basket/base year cost of base year basket to obtain the correct answer of 1.367.  Most of them are also able to explain that the inflation rate is inaccurate to measure cost of living changes because of substitution bias and quality adjustment bias. 

3(b) is a question on unemployment and most students also did well.  Most students correctly computed labour force (employed + unemployed), participation rate (labour force/adult population x 100%) and unemployment rate (unemployed/labour force x 100%).  However, some students computed the statistics wrongly, especially for unemployed which should be 200.

3(c) is a question on saving and investment and this question is well attempted.  Most students are able to draw the saving-investment diagram and illustrate that the investment demand curve shifts right with the investment tax credit.  The final effect is a higher real interest rate and a larger quantity of saving and investment. 

3(d) is a question on money creation and this question is well attempted.  Using the formula currency + (reserve/reserve-deposit ratio) = money supply, most students are able to obtain the correct answer of currency and bank reserves = $10,000.

Question 4

4(a) is a Keyensian model question using algebra.  Part (i) is not well attempted.  Most students did not realize that they need to set 40000 = PAE = C + IP + G + NX to solve for the real interest rate to get the correct real interest rate of 0.06 or 6%.  The performance of Part (ii) is better as most students know how to substitute r = 0.05 to obtain the PAE equation of 20,600 + 0.5Y and then equate with Y to solve for the equilibrium output which is 41,200 and establish the expansionary gap of 1,200.  Most students also know how to compute the spending multiplier of 2 and tax multiplier of -1 to obtain the decrease in government spending and increase in taxes to close the output gap.

4(b) is an AD-AS model question and this question is well attempted.  For Part (i), most students illustrate the decrease in export by a left shift of the AD curve in the short run but some student did not consider the rightward shift of AS curve in the long run to close the output gap using self-adjustment mechanism.  In Part (ii), most students know that central bank can increase money supply to shift the AD curve to the right to close the output gap.     

4(c) is a question on saving-investment in an open economy and the performance is mixed.  Some students did not consider the net capital inflow curve in the diagram together with the saving supply curve.  With budget deficit increases, the S+KI curve shifts left and the result is a higher real interest rate and a lower saving and investment quantity.
"""

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(vector_store_path)
    return db

def create_db_from_files():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(doc)
    print("chunking succesfully")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_store_path)
    print("saving succesfully")
    return db

def create_db_from_web():
    urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("chunking succesfully")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(doc_splits, embeddings)
    db.save_local(vector_store_path)
    print("saving succesfully")
    return db

create_db_from_web()

