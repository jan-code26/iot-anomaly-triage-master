from fastapi import FastAPI

# Create the FastAPI application instance.
# The title and version appear in the auto-generated API documentation
# at localhost:8000/docs — which you'll use constantly for testing.
app = FastAPI(
    title="Don't Trust the Sensors — IoT Anomaly Triage",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    """
    A simple health check endpoint.
    This is the first thing Render will ping to confirm your app is running.
    If this returns 200 OK, the deployment succeeded.
    """
    return {"status": "ok", "message": "Sensor triage system is running"}
