from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, Dict, List
from pydantic import BaseModel, validator, EmailStr
from datetime import datetime, timedelta
import re
import secrets
import json
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from database import SessionLocal, engine, database
from models import Base, User as DBUser, Task as DBTask

# Create tables
Base.metadata.create_all(bind=engine)

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

# Add security middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Database dependency
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Pydantic models for validation
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

    @validator('username')
    def username_validator(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        return v

    @validator('password')
    def password_validator(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class TaskBase(BaseModel):
    text: str
    priority: Optional[str] = "medium"

    @validator('text')
    def text_validator(cls, v):
        if len(v) < 1 or len(v) > 500:
            raise ValueError('Task text must be between 1 and 500 characters')
        return v

    @validator('priority')
    def priority_validator(cls, v):
        if v not in ["easy", "medium", "hard"]:
            raise ValueError('Priority must be easy, medium, or hard')
        return v

class TaskCreate(TaskBase):
    pass

class Task(TaskBase):
    id: int
    created_at: datetime
    owner_id: int

    class Config:
        orm_mode = True

# User database (replace with real database in production)
USERS_DB = {}

def detect_priority(text):
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

# Authentication endpoints
@app.post("/register", response_model=User)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    db_user = db.query(DBUser).filter(DBUser.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    priority: Optional[str] = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(DBTask).filter(DBTask.owner_id == current_user.id)
    if priority and priority != 'all':
        query = query.filter(DBTask.priority == priority)
    tasks = query.all()
    
    stats = {
        'total': db.query(DBTask).filter(DBTask.owner_id == current_user.id).count(),
        'easy': db.query(DBTask).filter(DBTask.owner_id == current_user.id, DBTask.priority == 'easy').count(),
        'medium': db.query(DBTask).filter(DBTask.owner_id == current_user.id, DBTask.priority == 'medium').count(),
        'hard': db.query(DBTask).filter(DBTask.owner_id == current_user.id, DBTask.priority == 'hard').count()
    }
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": tasks,
            "stats": stats,
            "current_filter": priority or 'all',
            "username": current_user.username
        }
    )

@app.post("/add", response_class=RedirectResponse)
async def add(
    task: TaskCreate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_task = DBTask(
        text=task.text,
        priority=detect_priority(task.text),
        owner_id=current_user.id
    )
    db.add(db_task)
    db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.post("/delete/{task_id}", response_class=RedirectResponse)
async def delete(
    task_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    task = db.query(DBTask).filter(
        DBTask.id == task_id,
        DBTask.owner_id == current_user.id
    ).first()
    
    if task:
        db.delete(task)
        db.commit()
    
    return RedirectResponse(url="/", status_code=303)

@app.post("/clear-all", response_class=RedirectResponse)
async def clear_all(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db.query(DBTask).filter(DBTask.owner_id == current_user.id).delete()
    db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/export")
async def export_tasks(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tasks = db.query(DBTask).filter(DBTask.owner_id == current_user.id).all()
    return JSONResponse(
        content=[{
            "text": task.text,
            "priority": task.priority,
            "created_at": task.created_at.isoformat()
        } for task in tasks],
        headers={
            "Content-Disposition": f"attachment; filename=tasks_{current_user.username}.json"
        }
    )

@app.post("/import", response_class=RedirectResponse)
async def import_tasks(
    file: UploadFile = File(...),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.json'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JSON files are allowed"
        )
    
    try:
        contents = await file.read()
        tasks_data = json.loads(contents)
        if not isinstance(tasks_data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )
        
        # Delete existing tasks
        db.query(DBTask).filter(DBTask.owner_id == current_user.id).delete()
        
        # Add new tasks
        for task_data in tasks_data:
            if not isinstance(task_data, dict) or 'text' not in task_data:
                continue
                
            # Sanitize and validate task text
            text = re.sub(r'[^\w\s!?]', '', task_data['text'])
            if len(text) > 500:
                text = text[:500]
                
            db_task = DBTask(
                text=text,
                priority=task_data.get('priority', 'medium'),
                owner_id=current_user.id
            )
            db.add(db_task)
            
        db.commit()
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON file"
        )
    
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
