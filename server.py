# %%writefile server.py
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import shutil
import json
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from map_logic import generate_map

baza = declarative_base()
engine = create_engine("sqlite:///observations.db")
Session = sessionmaker(bind=engine)
class Observation(baza):
    __tablename__ = "observations"
    id = Column(Integer, primary_key=True)
    lat = Column(Float)
    lon = Column(Float)
    text = Column(String)
    photo = Column(String)
    status = Column(String, default="red")
    created_at = Column(DateTime, default=datetime.utcnow)

class Route(baza):
    __tablename__ = "routes"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
baza.metadata.create_all(engine)
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("photos", exist_ok=True)
app.mount("/photos", StaticFiles(directory="photos"), name="photos")
class GenerateRequest(BaseModel):
    zhiv: str
    khar: str
    lat: float
    lon: float
    rad: float
    vol: int

@app.post("/generate")
def generate(req: GenerateRequest):
    result = generate_map(
        req.zhiv,
        req.khar,
        req.lat,
        req.lon,
        req.rad,
        req.vol
    )
    return result
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="templates", html=True), name="static")
@app.get("/observations")
def get_observations():
    db = Session()
    obs = db.query(Observation).all()
    return [
        {
            "id": o.id,
            "lat": o.lat,
            "lon": o.lon,
            "text": o.text,
            "status": o.status,
            "photo_url": f"/photos/{o.photo}" if o.photo else None,
            "created_at": o.created_at.isoformat()
        }
        for o in obs
    ]
@app.post("/observations")
def add_observation(
    lat: float = Form(...),
    lon: float = Form(...),
    text: str = Form(""),
    status: str = Form("red"),
    photo: UploadFile = File(None)
):
    db = Session()
    try:
      filename = None
      if photo:
          filename = f"{datetime.utcnow().timestamp()}_{photo.filename}"
          with open(f"photos/{filename}", "wb") as f:
              shutil.copyfileobj(photo.file, f)

      obs = Observation(lat=lat, lon=lon, text=text, photo=filename, status=status)
      db.add(obs)
      db.commit()
      db.refresh(obs)
      return {
            "id": obs.id,
            "lat": obs.lat,
            "lon": obs.lon,
            "text": obs.text,
            "status": obs.status,
            "photo_url": f"/photos/{obs.photo}" if obs.photo else None,
            "created_at": obs.created_at.isoformat(),
        }
    finally:
      db.close()
    return {"status": "ok"}

@app.post("/routes")
async def save_route(name: str = Form(...), data: str = Form(...)):
    db = Session()
    try:
        try:
            parsed = json.loads(data)
        except:
            return {"error": "invalid JSON"}
        route = Route(name=name, data=json.dumps(parsed))
        db.add(route)
        db.commit()
        db.refresh(route)
        return {"id": route.id, "status": "saved"}
    finally:
        db.close()
@app.get("/routes/{route_id}")
def get_route(route_id: int):
    db = Session()
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            return {"error": "not found"}
        try:
            data = json.loads(route.data)
        except:
            data = []
        return {
            "id": route.id,
            "name": route.name,
            "data": data,
            "created_at": route.created_at.isoformat()
        }
    finally:
        db.close()
@app.get("/routes")
def list_routes():
    db = Session()
    try:
        routes = db.query(Route).order_by(Route.created_at.desc()).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "created_at": r.created_at.isoformat()
            }
            for r in routes
        ]
    finally:
        db.close()