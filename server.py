# server.py
import os, math, json, asyncio
from dotenv import load_dotenv
load_dotenv()

from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import httpx


# ---- OpenAI-compatible client (official OpenAI SDK style) ----
# If you use the official SDK:
#   pip install openai==1.* fastapi uvicorn httpx
# And set: export OPENAI_API_KEY=sk-...
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    USE_OPENAI = True
except Exception:
    client = None
    OPENAI_MODEL = ""
    USE_OPENAI = False

app = FastAPI(title="AI Assistant")

from fastapi.responses import FileResponse

@app.get("/")
def landing():
    return FileResponse(os.path.join("static", "landing.html"))


# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------
# Tooling
# --------------------------
async def tool_calculator(expression: str) -> str:
    """
    Safe calculator for + - * / ^ (pow) and parentheses only.
    """
    allowed = "0123456789+-*/(). ^"
    if any(c not in allowed for c in expression):
        return "Error: Unsupported characters."
    try:
        # Replace '^' with '**'
        expr = expression.replace("^", "**")
        # Evaluate with limited builtins
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

async def tool_fetch(url: str) -> str:
    """
    Minimal page fetcher (GET). Returns first ~2000 chars of text.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as s:
            r = await s.get(url)
            text = r.text
            # Naive strip; for production, parse with readability.
            return text[:2000]
    except Exception as e:
        return f"Fetch error: {e}"

TOOLS_SPEC = {
    "calculator": {
        "description": "Evaluate a math expression like '2*(3+4)^2 / 7'.",
        "schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        "fn": tool_calculator,
    },
    "fetch": {
        "description": "Fetch the contents of a public webpage URL (GET). Returns plain text.",
        "schema": {"type": "object", "properties": {"url": {"type": "string", "format": "uri"}}, "required": ["url"]},
        "fn": tool_fetch,
    },
}

SYSTEM_PROMPT = """You are BizAssist AI — a direct, efficient business automation assistant designed for local service businesses (barbers, salons, cafés, restaurants, tattoo shops, gyms, nail studios, mechanics, freelancers, and other small service providers).

Your personality:
- Direct and efficient
- No unnecessary words
- Clear, actionable outputs
- Professional but not overly formal
- Always focused on saving the business owner time

Your 3 core responsibilities:

1) AUTOMATED CUSTOMER MESSAGE REPLIES
• Answer customer questions quickly and clearly.
• Provide prices, opening hours, locations, booking info, and service details.
• If the user needs custom info (like their business hours), ask for it once and then store it mentally for future replies in the conversation.
• Format messages cleanly so they can be copied into Instagram DMs, WhatsApp, or email.

2) SOCIAL MEDIA CONTENT GENERATION
• Generate Instagram captions, story ideas, weekly content plans, and promotional posts.
• Keep captions short, catchy, and optimized for local businesses.
• Include relevant hashtags when appropriate.
• Make content easy to copy and paste.

3) DOCUMENTS: INVOICES, ESTIMATES, TEMPLATES
• Generate invoice text, estimates, business templates, service descriptions, menus, and pricing lists.
• Provide clear, structured output.
• Ask for missing details (name, amount, service) once, then produce a clean template.

General rules:
• Never mention that you are an AI language model.
• Always act as a professional business assistant.
• Keep responses short unless the user asks for long output.
• Make everything ready-to-copy for real business use.
• If unsure, ask a short clarifying question.
- You must NEVER generate illegal, harmful, dangerous, threatening, blackmailing, hateful, or inappropriate content.
- You must NEVER assist in crimes, violence, hacking, scams, or anything unsafe.
- If a user requests something illegal or harmful, politely refuse and offer a safe, legal alternative.
- Never output personal data of real people unless the user provides it.
- Never produce medical, legal, or financial instructions that could harm someone.
- Always stay calm, polite, professional, and business-focused.


Your goal: Save the business owner time, increase their productivity, and automate as many repetitive tasks as possible.
"""

# --------------------------
# Schemas
# --------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    name: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

# --------------------------
# OpenAI Chat wrapper (streaming)
# --------------------------
async def openai_stream(messages: list[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not USE_OPENAI:
        yield "Server not configured with OPENAI_API_KEY.\n"
        return
    # Tool definitions for function-calling style
    tools = [{
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["schema"],
        }
    } for name, spec in TOOLS_SPEC.items()]

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
        stream=True,
    )
    tool_calls_buffer: list[Dict[str, Any]] = []
    assistant_text_chunks = []

    for event in stream:
        if event.choices and event.choices[0].delta:
            delta = event.choices[0].delta
            # Tool call?
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    # Accumulate tool call partials
                    if tc.type == "function":
                        tool_calls_buffer.append({
                            "id": tc.id,
                            "name": tc.function.name if tc.function else "",
                            "args_json": tc.function.arguments if tc.function else "{}",
                        })
            # Normal text
            if getattr(delta, "content", None):
                chunk = delta.content
                assistant_text_chunks.append(chunk)
                yield chunk

    # If there were tool calls, execute and send follow-up
    if tool_calls_buffer:
        tool_msgs = []
        for tc in tool_calls_buffer:
            name = tc["name"]
            args = {}
            try:
                args = json.loads(tc["args_json"] or "{}")
            except Exception:
                pass

            if name in TOOLS_SPEC:
                result = await TOOLS_SPEC[name]["fn"](**args)
            else:
                result = f"Tool '{name}' not found."
            tool_msgs.append({"role": "tool", "name": name, "content": result})

        # Send a second turn including tool results
        follow_messages = messages + [{"role": "assistant", "content": ""}] + tool_msgs
        stream2 = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=follow_messages,
            temperature=0.2,
            stream=True,
        )
        yield "\n\n"  # spacing
        for event in stream2:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content

# --------------------------
# Routes
# --------------------------
# --- add these imports near your other imports (keep only if not already present) ---
from pathlib import Path
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
# -------------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Build message list with system prompt
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]\

    # Load business profile and inject it (if it exists)
    if os.path.exists("business_profile.json"):
        with open("business_profile.json", "r") as f:
            profile = json.load(f)

        profile_text = (
            f"Business Name: {profile.get('business_name', '')}\n"
            f"Services: {profile.get('services', '')}\n"
            f"Pricing: {profile.get('pricing', '')}\n"
            f"Brand Voice: {profile.get('brand_voice', '')}\n"
            f"FAQs: {profile.get('faqs', '')}\n"
            f"Goals: {profile.get('goals', '')}\n"
        )

        msgs.append({
            "role": "system",
            "content": "Business Profile:\n" + profile_text
        })

    # Add user messages
    for m in req.messages:
        entry = {"role": m.role, "content": m.content}
        if m.role == "tool":
            entry["name"] = m.name or "tool"
        msgs.append(entry)

    # Stream the OpenAI response back to the browser
    return StreamingResponse(openai_stream(msgs), media_type="text/plain")



@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

# -------------------------------
# Business Profile Setup
# -------------------------------

from pydantic import BaseModel
import json
import os

PROFILE_PATH = "business_profile.json"

class BusinessProfile(BaseModel):
    business_name: str
    services: str
    pricing: str
    brand_voice: str
    faqs: str
    goals: str

@app.post("/business-profile")
def save_business_profile(profile: BusinessProfile):
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile.dict(), f, indent=4)
    return {"message": "Business profile saved successfully!"}

@app.get("/business-profile")
def get_business_profile():
    if not os.path.exists(PROFILE_PATH):
        return {"profile": None}
    with open(PROFILE_PATH, "r") as f:
        data = json.load(f)
    return {"profile": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    