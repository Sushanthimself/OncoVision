# OncoVision AI

Automated cancer cell detection and histopathological staging platform powered by multimodal LLM inference via Google Gemini 2.5 Flash.

> Built for the **Biology for Engineering** course.

---

## Architecture

```
[Next.js Frontend] ──(Multipart POST)──▶ [FastAPI Server] ──(PIL + Prompt)──▶ [Gemini 2.5 Flash]
                                                                                      │
                                                   ◀──(Structured Diagnostic JSON)────┘
```

## Repository Structure

```
OncoVision/
├── backend/
│   ├── .env.example          # Environment variable template
│   ├── main.py               # FastAPI server — /predict endpoint
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── public/               # Static assets
│   ├── src/
│   │   └── app/
│   │       ├── globals.css   # Design system — stark minimalist tokens
│   │       ├── layout.tsx    # Root layout with Geist fonts + SEO metadata
│   │       └── page.tsx      # Diagnostic Dashboard client component
│   ├── next.config.ts        # Next.js configuration
│   ├── package.json          # Node dependencies
│   ├── tailwind.config.ts    # Tailwind configuration
│   └── tsconfig.json         # TypeScript configuration
├── .gitignore
└── README.md
```

## Tech Stack

| Layer        | Technology                  |
| ------------ | --------------------------- |
| Frontend     | Next.js 15 (App Router)     |
| Styling      | Tailwind CSS 4              |
| Backend      | FastAPI (Python 3.10+)      |
| AI Engine    | Google GenAI SDK — gemini-2.5-flash |

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your_key_here"

# Run the server
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Contract

### `POST /predict`

**Request:** Multipart form with `file` field (JPEG/PNG/WebP/TIFF biopsy image).

**Response:**
```json
{
  "prediction": "Malignant",
  "confidence": 0.87,
  "biological_indicators": {
    "nc_ratio": "High",
    "pleomorphism": "Observed",
    "hyperchromasia": "Detected"
  },
  "reasoning": "The tissue sample exhibits significantly elevated nuclear-to-cytoplasmic ratios with irregular nuclear contours consistent with high-grade dysplasia. Dense hyperchromatic nuclei and marked pleomorphism across the cell population strongly suggest malignant transformation."
}
```

## License

Academic use — Biology for Engineering coursework.
