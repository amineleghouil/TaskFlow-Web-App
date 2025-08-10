from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import json
from datetime import datetime
import re
from typing import Optional, List
import os

app = FastAPI(title="Task Manager")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data validation models
class TaskBase(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Task text cannot be empty')
        if len(v) > 500:
            raise ValueError('Task text must be less than 500 characters')
        return v.strip()

class TaskCreate(TaskBase):
    pass

class Task(TaskBase):
    priority: str
    created_at: str

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

TASKS_FILE = 'tasks.json'

def detect_priority(text: str) -> str:
    # Keywords that indicate task complexity
    hard_keywords = [
        'urgent', 'asap', 'critical', 'complex', 'difficult', 'challenging',
        'important', 'crucial', 'emergency', 'deadline', 'priority'
    ]
    
    easy_keywords = [
        'simple', 'quick', 'easy', 'basic', 'small', 'minor', 'trivial',
        'fast', 'brief', 'short'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count words in the task
    word_count = len(text.split())
    
    # Check for time indicators (e.g., "2 hours", "3 days")
    time_pattern = r'\d+\s*(hour|day|week|month)'
    time_indicators = re.findall(time_pattern, text_lower)
    
    # Calculate scores
    hard_score = sum(1 for word in hard_keywords if word in text_lower)
    easy_score = sum(1 for word in easy_keywords if word in text_lower)
    
    # Additional complexity factors
    if word_count > 10:  # Longer tasks tend to be more complex
        hard_score += 1
    if time_indicators:  # Tasks with specific time requirements tend to be more important
        hard_score += 1
    if any(char in text for char in ['!', '?']):  # Exclamation or question marks might indicate urgency
        hard_score += 1
    
    # Determine priority based on scores
    if hard_score > easy_score:
        return 'hard'
    elif easy_score > hard_score:
        return 'easy'
    else:
        return 'medium'  # Default to medium if no clear indicators

def load_tasks():
    try:
        if not os.path.exists(TASKS_FILE):
            save_tasks([])
            return []
            
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
            if not isinstance(tasks, list):
                raise ValueError("Invalid tasks format")
            
            # Validate and sanitize existing tasks
            validated_tasks = []
            for task in tasks:
                if isinstance(task, dict) and 'text' in task and 'priority' in task:
                    # Sanitize text
                    task['text'] = re.sub(r'[^\w\s!?,.-]', '', task['text'])[:500]
                    # Validate priority
                    if task['priority'] not in ['easy', 'medium', 'hard']:
                        task['priority'] = 'medium'
                    validated_tasks.append(task)
            
            return validated_tasks
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"Error loading tasks: {str(e)}")
        return []

def save_tasks(tasks: List[dict]):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(TASKS_FILE) or '.', exist_ok=True)
        
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save tasks: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, priority: Optional[str] = None):
    try:
        tasks = load_tasks()
        
        # Validate priority parameter
        if priority and priority not in ['all', 'easy', 'medium', 'hard']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid priority filter"
            )
        
        # Calculate statistics
        stats = {
            'total': len(tasks),
            'easy': len([t for t in tasks if t['priority'] == 'easy']),
            'medium': len([t for t in tasks if t['priority'] == 'medium']),
            'hard': len([t for t in tasks if t['priority'] == 'hard'])
        }
        
        # Filter tasks if priority is specified
        if priority and priority != 'all':
            tasks = [t for t in tasks if t['priority'] == priority]
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "tasks": tasks,
                "stats": stats,
                "current_filter": priority or 'all'
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "tasks": [],
                "stats": {"total": 0, "easy": 0, "medium": 0, "hard": 0},
                "current_filter": "all",
                "error": str(e)
            }
        )

@app.post("/add")
async def add(task: str = Form(...)):
    try:
        # Validate task
        if not task or not task.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task cannot be empty"
            )
            
        # Sanitize input
        task = re.sub(r'[^\w\s!?,.-]', '', task.strip())[:500]
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task contains no valid characters"
            )
            
        priority = detect_priority(task)
        tasks = load_tasks()
        new_task = {
            'text': task,
            'priority': priority,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        tasks.append(new_task)
        save_tasks(tasks)
        
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/delete/{index}")
async def delete(index: int):
    try:
        tasks = load_tasks()
        if not (0 <= index < len(tasks)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        tasks.pop(index)
        save_tasks(tasks)
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/clear-all")
async def clear_all():
    try:
        save_tasks([])
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/export")
async def export_tasks():
    try:
        tasks = load_tasks()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return JSONResponse(
            content=tasks,
            headers={
                "Content-Disposition": f"attachment; filename=tasks_{timestamp}.json",
                "Content-Type": "application/json"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/import")
async def import_tasks(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JSON files are allowed"
            )
        
        if file.content_type != 'application/json':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Must be JSON"
            )
            
        # Limit file size to 1MB
        MAX_SIZE = 1 * 1024 * 1024  # 1MB
        contents = b""
        size = 0
        
        async for chunk in file.stream():
            size += len(chunk)
            if size > MAX_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large. Maximum size is 1MB"
                )
            contents += chunk
        
        try:
            tasks = json.loads(contents)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )
            
        if not isinstance(tasks, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid format. Expected a list of tasks"
            )
        
        # Validate and sanitize imported tasks
        validated_tasks = []
        for task in tasks:
            if not isinstance(task, dict) or 'text' not in task:
                continue
                
            # Sanitize text
            text = re.sub(r'[^\w\s!?,.-]', '', task['text'].strip())[:500]
            if not text:
                continue
                
            # Validate or assign priority
            priority = task.get('priority', 'medium')
            if priority not in ['easy', 'medium', 'hard']:
                priority = detect_priority(text)
                
            validated_tasks.append({
                'text': text,
                'priority': priority,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        
        if not validated_tasks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid tasks found in the file"
            )
            
        save_tasks(validated_tasks)
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=None,  # Add your SSL key file path for HTTPS
        ssl_certfile=None  # Add your SSL cert file path for HTTPS
    )
