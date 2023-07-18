from flask import Flask, request
from pymessenger.bot import Bot
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
import json

app = Flask(__name__)
ACCESS_TOKEN = 'EAANIqNj4YJQBABEC3qkGKpfHZBtZB0pU2s0PjhjPhePTESy1ZAgOwqXpKDzeEU2FuEz8QEh9lOpQA6DPdHSbKw8d5kKtga6nsGFVb8MvhpDq6JqlRbmePDDgEyHZBM0bPRg3gT4eDGewXfiuFrlZCnaTGHwnKB9PlDz3srUBXN3ZCtGIX6fHqJ'
VERIFY_TOKEN = 'Hoover'
bot = Bot(ACCESS_TOKEN)
DOC_LOCATION = 'MyWork Knowledge Base.txt'
prompt_template = "You are an ethusiastic customer support and sales representative for the" \
                  " company described in the context.\nYou are to answer the" \
                  " customers question to the best of your ability, keeping" \
                  " responses capped at 50 words maximium.\nIf you don't know" \
                  " the answer just say you dont know and refer the customer" \
                  " to the businesses contact information,\ndont make up an" \
                  " answer. Frame every response in a way that makes the" \
                  " business look like the best solution for the customers problem. Refer to the company as 'we'.\n" \
                  " Try to motivate the customer to use the company as their service provider. If a quote is mentioned always provide direct contact information\n" \
                  "-----------\n" \
                  "{context}\n"

OPENAI_API_KEY = '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'
model_name = 'text-embedding-ada-002'
PINECONE_KEY = 'cff77a36-2b36-40e9-a286-7a3d671a63e3'
PINECORN_ENV = 'asia-southeast1-gcp-free'
INDEX_NAME = 'langchain-retrieval-augmentation'
tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


pinecone.init(
    api_key=PINECONE_KEY,
    environment=PINECORN_ENV
)

index = pinecone.Index(INDEX_NAME)

text_field = "text"

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.5
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
qa.combine_documents_chain.llm_chain.prompt.messages[
    0].prompt.template = prompt_template

data_source = Tool(
    name='Knowledge Base',
    func=qa.run,
    description=(
        'use this tool when answering general knowledge queries to get '
        'more information about the topic'
    )
)

tools = [data_source]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=False,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

def single_query(query):
    return qa.run(query)

def continous_query():
    query = input("What can I help you with? ")
    while query != '' and query is not None:
        print(agent(query)['output'])
        query = input("Anything else? ")



@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    else:
        output = request.get_json()
        for event in output['entry']:
            if event['messaging']:
                messaging = event['messaging']
            else: break
            for message in messaging:
                if message['message'].get('text') and message['message'].get('is_echo') is None:
                    recipient_id = message['sender']['id']
                    typing_payload = json.dumps(
                        {"recipient": {"id": recipient_id},
                         "sender_action": "mark_seen"})
                    print(typing_payload)
                    print(bot.send_raw(typing_payload))
                    query = message['message'].get('text')
                    response_sent_text = single_query(query)
                    send_message(recipient_id, response_sent_text)
                    if message['message'].get('attachments'):
                        response_sent_nontext = None
                        send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)
    return "success"


if __name__ == "__main__":
    app.run(port=8080)
