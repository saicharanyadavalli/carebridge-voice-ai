import os
import uvicorn
from app.config import config

if __name__ == "__main__":
    host = os.getenv("HOST", config.HOST)
    port = int(os.getenv("PORT", str(config.PORT)))
    print(f"Starting CareBridge on http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
