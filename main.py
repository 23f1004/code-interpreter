import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN", "missing"),
    base_url="https://aipipe.org/openai/v1",
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---- Tool: runs the code ----
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except BaseException:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


# ---- Helper: line number straight from the traceback ----
def line_from_traceback(tb: str) -> List[int]:
    matches = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(matches[-1])] if matches else []


# ---- AI agent: finds the error line ----
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    tb_lines = line_from_traceback(tb)
    total_lines = len(code.splitlines())

    numbered = "\n".join(
        f"{i}: {line}" for i, line in enumerate(code.splitlines(), start=1)
    )
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.
Line numbers start at 1 and refer to the CODE below.
The traceback line 'File "<string>", line N' tells you the line number.

CODE (with line numbers):
{numbered}

TRACEBACK:
{tb}

Reply ONLY with JSON in this exact shape: {{"error_lines": [3]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        ai_lines = ErrorAnalysis.model_validate_json(text).error_lines
    except Exception:
        return tb_lines

    # Safety check: trust the traceback if the AI disagrees or is out of range
    if tb_lines and ai_lines != tb_lines:
        return tb_lines
    if not ai_lines or any(n < 1 or n > max(total_lines, 1) for n in ai_lines):
        return tb_lines
    return ai_lines


@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    result = execute_python_code(req.code)
    if result["success"]:
        lines = []
    else:
        lines = analyze_error_with_ai(req.code, result["output"])

    print("=== NEW REQUEST ===")
    print("CODE:", repr(req.code))
    print("OUTPUT:", result["output"])
    print("LINES RETURNED:", lines)
    print("===================", flush=True)

    return {"error": lines, "result": result["output"]}