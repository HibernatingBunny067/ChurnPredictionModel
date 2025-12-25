#!/bin/bash
echo "Training models..."
python -m src.pipeline.train_pipeline

echo "Starting FastAPI Backend..."
python app.py &

sleep 5

echo "Starting Streamlit Frontend..."
streamlit run demo.py --server.port 8501 --server.address 0.0.0.0