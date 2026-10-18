"""
Quick verification script to test Google Gemini integration for LangGraph / LangChain Academy.
Supports both Google AI Studio (GEMINI_API_KEY) and GCP Vertex AI.
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress LangSmith if using placeholder
if os.environ.get("LANGSMITH_API_KEY", "").startswith("your-"):
    os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ["LANGSMITH_TRACING"] = "false"

def test_ai_studio():
    api_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    # Filter out dummy placeholder strings
    if api_key and api_key.startswith("your-"):
        api_key = None
    if google_key and google_key.startswith("your-"):
        google_key = None

    chosen_key = api_key or google_key
    if not chosen_key:
        print("[-] Valid GEMINI_API_KEY / GOOGLE_API_KEY not found in environment or .env file.")
        return False

    # Ensure langchain-google-genai uses the valid key
    os.environ["GOOGLE_API_KEY"] = chosen_key
    os.environ["GEMINI_API_KEY"] = chosen_key

    print("[*] Testing Google AI Studio (langchain-google-genai)...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
        res = model.invoke([HumanMessage(content="Say 'Gemini connection successful!' and nothing else.")])
        # Format response cleanly if returned as list or text
        content = res.content
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            content = content[0].get("text", content)
        print(f"[+] Response: {content.strip() if isinstance(content, str) else content}")
        return True
    except Exception as e:
        print(f"[-] AI Studio test failed: {e}")
        return False

def test_vertex_ai():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project and project.startswith("your-"):
        project = None
    if not project:
        print("[*] Skipping GCP Vertex AI (No GCP project configured; using AI Studio).")
        return False
    print(f"[*] Testing GCP Vertex AI (Project: {project})...")
    try:
        from langchain_google_vertexai import ChatVertexAI
        from langchain_core.messages import HumanMessage

        kwargs = {"model": "gemini-1.5-flash", "temperature": 0}
        if project:
            kwargs["project"] = project
        region = os.environ.get("GOOGLE_CLOUD_REGION")
        if region:
            kwargs["location"] = region

        model = ChatVertexAI(**kwargs)
        res = model.invoke([HumanMessage(content="Say 'Vertex AI connection successful!' and nothing else.")])
        print(f"[+] Response: {res.content.strip()}")
        return True
    except Exception as e:
        print(f"[-] Vertex AI test failed: {e}")
        return False

def main():
    print("=" * 60)
    print(" Google Gemini / GCP Connectivity Diagnostic")
    print("=" * 60)

    ai_studio_ok = test_ai_studio()
    print("-" * 60)
    vertex_ok = test_vertex_ai()
    print("=" * 60)

    if ai_studio_ok or vertex_ok:
        print("[SUCCESS] At least one Google Gemini integration is functioning!")
    else:
        print("[NOTICE] Configure either your GEMINI_API_KEY or GCP Vertex AI credentials to proceed.")

if __name__ == "__main__":
    main()
