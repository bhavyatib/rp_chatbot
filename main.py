import os
import time
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ==== CONFIG ====
API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

client = OpenAI(api_key=API_KEY)
app = FastAPI()

# Allow cross-origin (so your chat widget can call from any site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_citations(text):
    cleaned = re.sub(r"【\d+:\d+†[^\】]+】", "", text)
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned

# In-memory store for user threads (user_id: thread_id)
user_threads = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

assistant_id = None

@app.on_event("startup")
def startup_event():
    global assistant_id
    print("Creating assistant...")
    assistant = client.beta.assistants.create(
        name="Raghav Polymers Chatbot",
        instructions=(
            "You are a helpful and kind customer service executive for Raghav Polymers Group. "
            "Prefer answering from the company documents in the vector store. "
            "If not found, use your general knowledge."
        ),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}}
    )
    assistant_id = assistant.id
    print(f"Assistant ready: {assistant_id}")

@app.post("/chat")
def chat_endpoint(chat: ChatRequest):
    # Manage user thread
    user_id = chat.user_id
    if user_id not in user_threads:
        thread = client.beta.threads.create()
        user_threads[user_id] = thread.id
    thread_id = user_threads[user_id]

    # Add user message to thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=chat.message
    )

    # Run assistant
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # Wait for completion
    for _ in range(60):
        status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if status.status == "completed":
            break
        elif status.status == "failed":
            return {"answer": "Assistant failed to respond."}
        time.sleep(1)
    else:
        return {"answer": "Timeout waiting for response."}

    # Get assistant message for THIS run
    run_steps = client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run.id)
    assistant_message_id = None
    for step in run_steps.data:
        if hasattr(step, "step_details") and hasattr(step.step_details, "message_creation"):
            assistant_message_id = step.step_details.message_creation.message_id
            break

    if assistant_message_id:
        msg = client.beta.threads.messages.retrieve(
            thread_id=thread_id,
            message_id=assistant_message_id
        )
        answer = msg.content[0].text.value.strip()
        answer = clean_citations(answer)
        return {"answer": answer}
    else:
        return {"answer": "No answer found for this question."}
