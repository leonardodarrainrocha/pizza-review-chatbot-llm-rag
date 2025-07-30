import vector
import requests
import subprocess
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def getAvailableModelNames():
    availableModelNames = ""
    info = subprocess.run(["cmd", "/c", "ollama list"], capture_output=True, text=True).stdout.split("\n")[1:]
    for item in info:
        availableModelNames += item[0:item.find(":")] + " | "
    return availableModelNames[:-6]

def getModel(name:str):
    print("Loading...")
    template = """
    Your are an expert in answering question about a pizza restaurant
    Here are some relevant reviews: {reviews}
    Here is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=name)
    if name == "llama3.2":
        vector.createVectorDb(name)
    
    try:
        details = requests.post("http://localhost:11434/api/show", json={"name": name}).json()["details"]
        print(f"{details['family']} ({details['parameter_size']}) is Loaded")
    except Exception as e:
        print(f"{model.model} is Loaded")

    chain = prompt | model
    return chain

question = ""
while question != "q":
    availableModelNames = getAvailableModelNames()
    print(f"\nAvailable models -> {availableModelNames}")
    question = input("Select a model (q to quit): ").strip()
    if question == "q" or availableModelNames.find(question) == -1:
        continue
    chain = getModel(question)
    modelName = question
    
    while question != "m":
        question = input("\nAsk your question (m to change model, q to quit): ").strip()
        if question == "q":
            break
        if question == "m":
            continue
        print("\n\n")
        if modelName == "llama3.2":
            retriever = vector.getRetriever()
            reviews = retriever.invoke(question)
            result = chain.invoke({"reviews": reviews, "question": question})
        else:
            result = chain.invoke({"reviews": [], "question": question})
        print(result)
        print("\n\n")


