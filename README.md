# Name : Ankita Patra
# MS in Computer Science | Boston University

Focused on Backend Engineering, Cloud Systems & Applied ML
############################# Overview #############################

This project implements an end-to-end, production-ready sentiment analysis system that predicts whether a movie review expresses positive or negative sentiment.
The system goes beyond a basic ML notebook and demonstrates real-world ML engineering, including:
Modern NLP model fine-tuning
Cloud-based training & inference
A production-grade backend API
Performance, reliability, and cost-aware design
A simple web frontend that consumes the API
The project uses AWS SageMaker, Lambda, API Gateway, and PyTorch, and is deployed as a fully functional cloud service.

############################# Model & ML Approach #############################

Model: Transformer-based NLP model (DistilBERT)

Task: Binary sentiment classification (Positive / Negative)

Dataset: IMDb Movie Reviews

Training:
Fine-tuned on IMDb dataset
GPU-backed training on SageMaker
Optimized for inference latency and memory

Output:
Prediction label (0 = Negative, 1 = Positive)
Confidence probability


############################# System Architecture #############################
Browser (Web App)
      ↓
API Gateway (Public REST API)
      ↓
AWS Lambda (Auth, validation, routing)
      ↓
SageMaker Endpoint (ML inference)
      ↓
PyTorch Model (DistilBERT)

############################# Key Features #############################
 Modern ML Model

Transformer-based NLP (DistilBERT)
Dynamic padding & batch inference
GPU acceleration for training
Optimized inference using torch.inference_mode()
Production-Grade Backend API
JSON-based request/response contract
Input validation & schema enforcement
Batch inference support
Stable response format with latency metadata

Example Request

{
  "text": "This movie was fantastic!"
}


Example Response

{
  "predictions": [
    { "label": 1, "prob": 0.94 }
  ],
  "latency_ms": 38
}

############################# Performance & Reliability #############################

Batch tokenization for throughput
Dynamic padding (no wasted computation)
Timeout & error handling (frontend + backend)
Input size and batch-size limits
Graceful failure handling

############################# Cloud & Deployment #############################

SageMaker training jobs (GPU-backed)
SageMaker real-time inference endpoint
Lambda-based inference routing
API Gateway for public access
Static web frontend (HTML + JS)

############################# Security & Cost Awareness #############################

No AWS credentials exposed to frontend
API Gateway as controlled entry point
Lambda IAM role with scoped permissions
Input size limits to prevent abuse
Endpoints shut down when not in use

############################# Project Structure #############################
.
├── train/
│   ├── train.py          # SageMaker training script
│   ├── model.py          # PyTorch model definition
│   └── requirements.txt
│
├── serve/
│   ├── predict.py        # Production inference logic
│   ├── model.py
│   ├── utils.py
│   └── requirements.txt
│
├── website/
│   └── index.html        # Frontend web app
│
├── SageMaker_Project.ipynb
└── README.md

############################# Testing #############################

Automated testing via SageMaker endpoint
Manual testing via:
Notebook inference
Lambda + API Gateway
Web UI submission
Accuracy ~84–86% on IMDb test samples