# app.py
import os
import io
import json
import tempfile
import zipfile
import tarfile
import base64
import subprocess
import logging
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# load .env if present (useful for local dev)
load_dotenv()

# ---------- Configuration (set these via environment variables or .env) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")           # <-- put your key here in .env or env
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")        # optional (if you're using a proxy)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")      # default model; change as needed
GENERATED_CODE_TIMEOUT = int(os.getenv("GENERATED_CODE_TIMEOUT", "150"))  # seconds
# ------------------------------------------------------------------------------

if not OPENAI_API_KEY:
    # We do not crash here so devs can test the upload parsing without contacting OpenAI,
    # but the endpoint will return an error when it needs to call the LLM.
    logging.warning("OPENAI_API_KEY is not set. Set it in your environment or in a .env file.")

app = FastAPI()

# Small helper dataclass-like object
class FileData:
    def __init__(self, name: str, content: bytes, content_type: str, is_image=False, is_text=False):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.is_image = is_image
        self.is_text = is_text

def get_content_type_for_image(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "gif":
        return "image/gif"
    return "application/octet-stream"

def strip_code_fence(s: str) -> str:
    """Remove triple-backtick fences if the model returned code inside fences."""
    if s.strip().startswith("```") and s.strip().endswith("```"):
        # drop outer fences
        parts = s.strip().split("\n")
        # remove first line (```[lang]) and last line (```)
        return "\n".join(parts[1:-1])
    return s

@app.post("/api/")
async def analyze_data(request: Request):
    """
    Accepts a multipart/form-data POST with:
      - an uploaded file named 'question.txt' (or 'questions.txt')
      - optional supporting files (zip/tar or single files)
    Workflow:
      - parse files and build a textual prompt
      - call LLM to generate Python code that prints JSON to stdout
      - save code, run it in a subprocess (timeout), capture stdout
      - parse stdout as JSON and return it directly
    """
    form = await request.form()

    # ------------- Find question file (accept question.txt or questions.txt) -------------
    question_file = None
    for v in form.values():
        if hasattr(v, "filename") and v.filename:
            fn = v.filename.lower()
            if fn == "question.txt" or fn == "questions.txt":
                question_file = v
                break

    if question_file is None:
        return JSONResponse(status_code=400, content={"error": "Missing required file named 'question.txt' (or 'questions.txt')."})

    # ------------- Read question text and other uploaded files -------------
    questions_content = (await question_file.read()).decode("utf-8", errors="replace")
    processed_files: List[FileData] = []

    # Create a temp dir to extract archives safely
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, file_or_field in form.items():
            if not hasattr(file_or_field, "filename"):
                continue
            uploaded: UploadFile = file_or_field
            if uploaded.filename.lower() in ("question.txt", "questions.txt"):
                continue  # already read

            content = await uploaded.read()
            is_zip = uploaded.filename.lower().endswith(".zip")
            is_tar = uploaded.filename.lower().endswith((".tar", ".tar.gz", ".tgz"))

            if is_zip or is_tar:
                # Extract archive into temp_dir and add each extracted file
                archive_bytes = io.BytesIO(content)
                try:
                    if is_zip:
                        with zipfile.ZipFile(archive_bytes) as zf:
                            zf.extractall(temp_dir)
                    else:
                        # tarfile
                        archive_bytes.seek(0)
                        with tarfile.open(fileobj=archive_bytes) as tf:
                            tf.extractall(temp_dir)

                    # add extracted files
                    for root, dirs, files in os.walk(temp_dir):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            with open(fpath, "rb") as fh:
                                fcont = fh.read()
                            is_image = fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                            content_type = get_content_type_for_image(fname) if is_image else "text/plain"
                            processed_files.append(FileData(name=fname, content=fcont, content_type=content_type, is_image=is_image, is_text=not is_image))
                except Exception as e:
                    # treat archive load errors as a single uploaded file error
                    return JSONResponse(status_code=400, content={"error": f"Could not extract archive {uploaded.filename}: {str(e)}"})
            else:
                is_image = uploaded.content_type and uploaded.content_type.startswith("image/")
                processed_files.append(FileData(name=uploaded.filename, content=content, content_type=uploaded.content_type or "application/octet-stream", is_image=is_image, is_text=not is_image))

    # ------------- Build a compact prompt text (text only) -------------
    # NOTE: we send only text (and small previews). If you need to send large binary images, use a
    # separate image upload + URL; embedding base64 into prompt can exceed model limits.
    prompt_parts = []
    prompt_parts.append("User question:")
    prompt_parts.append(questions_content.strip())
    prompt_parts.append("\nFiles attached (previews):")

    for f in processed_files:
        if f.is_image:
            # we provide filename and content-type — not embedding full bytes by default
            prompt_parts.append(f"- Image file: {f.name} (content-type: {f.content_type})")
        else:
            # text preview
            try:
                txt = f.content.decode("utf-8", errors="replace")
                preview = "\n".join(txt.splitlines()[:40])
                prompt_parts.append(f"- Text file: {f.name}\n---preview---\n{preview}\n---end-preview---")
            except Exception:
                prompt_parts.append(f"- Binary file: {f.name} (binary)")

    prompt_text = "\n\n".join(prompt_parts)

    # ---------- Call the LLM to generate Python code ----------
    # We expect the model to output only Python code that when executed prints valid JSON to stdout.
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OPENAI_API_KEY is not set in environment."})

    # Import OpenAI at call-time to avoid failing earlier if package missing
    try:
        import openai
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"openai package not installed: {e}"})

    # configure client (works with openai-python v1+)
    try:
        openai.api_key = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            openai.api_base = OPENAI_BASE_URL
    except Exception:
        pass

    system_prompt = (
        "You are a production-level data analyst. Produce a single Python program as the ONLY output.\n"
        "REQUIREMENTS:\n"
        " - The program must be pure Python (assume pandas/numpy/matplotlib are available).\n"
        " - The program must print exactly one JSON value to stdout (an array or object) with the answers requested.\n"
        " - If producing an image, save it to a file and print its base64 data URI as part of the JSON.\n"
        " - Do NOT output explanatory text. The HTTP service will run the program and parse stdout as JSON.\n"
    )

    # Compose messages for chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]

    # call ChatCompletion
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=3000
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"LLM call failed: {str(e)}"})

    # extract model content (robust to different SDK response structures)
    try:
        generated = response.choices[0].message["content"]
    except Exception:
        try:
            generated = response.choices[0].text
        except Exception:
            generated = str(response)

    generated_code = strip_code_fence(generated)

    # ---------- Save generated code and execute it safely in subprocess ----------
    # WARNING: Executing model-generated code is risky. Only do this in a sandbox (Docker) in production.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmpf:
        tmpf.write(generated_code)
        tmpfname = tmpf.name

    try:
        # run code in a subprocess; capture stdout and stderr
        proc = subprocess.run(
            ["python", tmpfname],
            capture_output=True,
            text=True,
            timeout=GENERATED_CODE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmpfname)
        return JSONResponse(status_code=504, content={"error": f"Generated code timed out after {GENERATED_CODE_TIMEOUT} seconds."})
    except Exception as e:
        os.unlink(tmpfname)
        return JSONResponse(status_code=500, content={"error": f"Failed to run generated code: {str(e)}"})

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    os.unlink(tmpfname)

    if proc.returncode != 0:
        # Execution failed; return helpful debug info (so you can fix prompt or model)
        return JSONResponse(status_code=500, content={"error": "Generated code returned non-zero exit code.", "stderr": stderr, "stdout": stdout})

    # Attempt to parse stdout as JSON (expected behavior)
    try:
        parsed = json.loads(stdout)
        # On success, return the parsed JSON directly (no wrappers)
        return JSONResponse(content=parsed)
    except Exception as e:
        # Not valid JSON — return debugging info
        return JSONResponse(status_code=500, content={
            "error": "Generated code did not print valid JSON.",
            "parse_error": str(e),
            "stdout": stdout,
            "stderr": stderr,
            "generated_code_preview": generated_code[:2000]
        })


if __name__ == "__main__":
    # Local dev: run server
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
