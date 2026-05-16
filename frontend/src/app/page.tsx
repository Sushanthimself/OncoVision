"use client";

import React, { useState, useRef, useCallback } from "react";

/* ─────────────────────────────────────────────────────────────────────────────
   TYPE DEFINITIONS
   ───────────────────────────────────────────────────────────────────────────── */

interface BiologicalIndicators {
  nc_ratio: "High" | "Normal";
  pleomorphism: "Observed" | "Not Observed";
  hyperchromasia: "Detected" | "Not Detected";
}

interface CaseAnalysis {
  "0_pathogenesis": string;
  "1_clinical_features": string;
  "2_radiographic_features": string;
  "3_histologic_features": string;
  "4_provisional_diagnosis": string;
  "5_treatment_planning": string;
  "6_potential_complications": string;
  "7_transformation_probability"?: string;
  "8_classification_percentage"?: string;
  [key: string]: string | undefined;
}

interface DiagnosisResult {
  prediction: string;
  confidence: number;
  biological_indicators: BiologicalIndicators;
  case_analysis: CaseAnalysis;
}

/* ─────────────────────────────────────────────────────────────────────────────
   CONSTANTS
   ───────────────────────────────────────────────────────────────────────────── */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* ─────────────────────────────────────────────────────────────────────────────
   PAGE COMPONENT
   ───────────────────────────────────────────────────────────────────────────── */

export default function DiagnosticDashboard() {
  /* state ------------------------------------------------------------------ */
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  /* handlers --------------------------------------------------------------- */

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setDiagnosis(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("image/")) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const onFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const resetState = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setDiagnosis(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const runInference = useCallback(async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setDiagnosis(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(
          body?.detail ?? `Server responded with status ${res.status}`
        );
      }

      const data: DiagnosisResult = await res.json();
      setDiagnosis(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile]);

  /* derived ---------------------------------------------------------------- */
  // Gemini may return confidence as 0-1 (e.g. 0.98) or 0-100 (e.g. 98).
  // Auto-detect: if the value is > 1, treat it as already a percentage.
  const rawConf = diagnosis?.confidence ?? 0;
  const confidencePct = rawConf > 1 ? Math.round(rawConf) : Math.round(rawConf * 100);

  // Gemini returns specific condition names (e.g. "Invasive Ductal Carcinoma")
  // not generic "Malignant"/"Benign" labels. Detect malignancy from the
  // biomarker signals: if at least 2 of 3 are abnormal, it is likely malignant.
  const detectMalignancy = (d: DiagnosisResult): boolean => {
    let abnormalCount = 0;
    if (d.biological_indicators.nc_ratio === "High") abnormalCount++;
    if (d.biological_indicators.pleomorphism === "Observed") abnormalCount++;
    if (d.biological_indicators.hyperchromasia === "Detected") abnormalCount++;
    return abnormalCount >= 2;
  };

  const isMalignant = diagnosis ? detectMalignancy(diagnosis) : false;

  /* layman summary --------------------------------------------------------- */
  const getLaymanSummary = (d: DiagnosisResult): { headline: string; bullets: string[] } => {
    const isMal = detectMalignancy(d);
    const confValue = d.confidence > 1 ? Math.round(d.confidence) : Math.round(d.confidence * 100);
    const bullets: string[] = [];

    // N:C ratio
    if (d.biological_indicators.nc_ratio === "High") {
      bullets.push("🔬 The cell nuclei (control centres) are unusually large compared to the rest of the cell — a classic early warning sign that something is off.");
    } else {
      bullets.push("🔬 Cell nuclei appear a normal size relative to the rest of the cell — no alarm there.");
    }

    // Pleomorphism
    if (d.biological_indicators.pleomorphism === "Observed") {
      bullets.push("📐 The cells are irregular and misshapen — healthy cells are uniform, so this variety in shape and size is a red flag.");
    } else {
      bullets.push("📐 The cells all look roughly the same size and shape — exactly what you want to see in healthy tissue.");
    }

    // Hyperchromasia
    if (d.biological_indicators.hyperchromasia === "Detected") {
      bullets.push("🎨 The nuclei appear abnormally dark under the microscope because the DNA is packed too densely — this is often linked to rapid, uncontrolled cell division.");
    } else {
      bullets.push("🎨 The nuclei stain a normal colour, meaning the DNA packing looks healthy and controlled.");
    }

    const headline = isMal
      ? `The AI detected signs consistent with ${d.prediction} with ${confValue}% confidence. The cells show abnormal behaviour that warrants further clinical investigation.`
      : `The AI found no significant signs of malignancy with ${confValue}% confidence. The tissue classified as "${d.prediction}" appears to be behaving normally based on the three markers examined.`;

    return { headline, bullets };
  };

  const layman = diagnosis ? getLaymanSummary(diagnosis) : null;

  /* render ----------------------------------------------------------------- */
  return (
    <div className="flex flex-col min-h-screen">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto max-w-6xl flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-black flex items-center justify-center">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="square"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v8M8 12h8" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-black leading-none">
                OncoVision AI
              </h1>
              <p className="text-[11px] text-zinc-400 tracking-wide uppercase mt-0.5">
                Histopathological Staging
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[11px] font-mono text-zinc-400 tracking-wider uppercase">
              Engine: Gemini 2.5 Flash
            </span>
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="text-[11px] text-zinc-500">Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <main className="flex-1 mx-auto max-w-6xl w-full px-6 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* ── Left Column — Upload ────────────────────────────────────────── */}
          <div className="flex flex-col gap-6">
            {/* Upload card */}
            <div className="bg-white border border-zinc-200 p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-widest text-black">
                    Biopsy Input
                  </h2>
                  <p className="text-[11px] text-zinc-400 mt-1">
                    Upload a stained histopathological tissue sample
                  </p>
                </div>
                {selectedFile && (
                  <button
                    onClick={resetState}
                    className="text-[11px] text-zinc-400 hover:text-black transition-colors uppercase tracking-wider"
                    id="btn-reset"
                  >
                    Clear
                  </button>
                )}
              </div>

              {/* Dropzone */}
              <div
                onDrop={onDrop}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  relative border border-dashed cursor-pointer
                  transition-all duration-200 aspect-square
                  flex items-center justify-center overflow-hidden
                  ${isDragOver ? "dropzone-active" : "border-zinc-300 hover:border-zinc-400 bg-zinc-50"}
                `}
                id="dropzone"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/tiff"
                  onChange={onFileInputChange}
                  className="hidden"
                  id="file-input"
                />

                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Biopsy preview"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-3 text-center px-6">
                    <div className="w-10 h-10 border border-zinc-300 flex items-center justify-center">
                      <svg
                        width="20"
                        height="20"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="#a1a1aa"
                        strokeWidth="1.5"
                        strokeLinecap="square"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs text-zinc-500">
                        Drag and drop or{" "}
                        <span className="text-black font-medium underline underline-offset-2">
                          browse files
                        </span>
                      </p>
                      <p className="text-[10px] text-zinc-400 mt-1">
                        JPEG · PNG · WebP · TIFF
                      </p>
                    </div>
                  </div>
                )}

                {/* Scan line overlay during loading */}
                {isLoading && (
                  <div className="absolute inset-0 bg-white/60 flex items-center justify-center">
                    <div className="absolute inset-0 overflow-hidden">
                      <div className="absolute inset-x-0 h-px bg-black animate-scan-line" />
                    </div>
                  </div>
                )}
              </div>

              {/* File meta */}
              {selectedFile && (
                <div className="mt-4 flex items-center justify-between text-[11px] text-zinc-400 font-mono">
                  <span className="truncate max-w-[60%]">{selectedFile.name}</span>
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                </div>
              )}
            </div>

            {/* Action button */}
            <button
              onClick={runInference}
              disabled={!selectedFile || isLoading}
              className={`
                w-full py-3.5 text-xs font-semibold uppercase tracking-[0.2em] transition-all duration-200
                ${
                  !selectedFile || isLoading
                    ? "bg-zinc-100 text-zinc-300 cursor-not-allowed"
                    : "bg-black text-white hover:bg-zinc-900 active:scale-[0.99]"
                }
              `}
              id="btn-analyze"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-3.5 w-3.5"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="3"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                  Processing Matrix…
                </span>
              ) : (
                "Run Diagnostic Analysis"
              )}
            </button>

            {/* Error banner */}
            {error && (
              <div className="bg-white border border-zinc-200 p-4 animate-fade-in-up" id="error-banner">
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 border border-zinc-300 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-[10px] font-bold text-black">!</span>
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-black">
                      Inference Error
                    </p>
                    <p className="text-xs text-zinc-500 mt-1 leading-relaxed">{error}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── Right Column — Results ──────────────────────────────────────── */}
          <div className="flex flex-col gap-6">
            {!diagnosis && !isLoading && (
              <div className="bg-white border border-zinc-200 h-full flex items-center justify-center min-h-[400px]">
                <div className="text-center px-8">
                  <div className="w-12 h-12 border border-zinc-200 flex items-center justify-center mx-auto mb-4">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#d4d4d8"
                      strokeWidth="1.5"
                      strokeLinecap="square"
                    >
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                      <line x1="16" y1="13" x2="8" y2="13" />
                      <line x1="16" y1="17" x2="8" y2="17" />
                    </svg>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    Diagnostic readout will render here
                    <br />
                    after model inference completes.
                  </p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="bg-white border border-zinc-200 p-6 min-h-[400px] flex flex-col gap-5 animate-pulse-slow">
                <div className="h-3 bg-zinc-100 w-1/3" />
                <div className="h-10 bg-zinc-100 w-full" />
                <div className="h-3 bg-zinc-100 w-2/3" />
                <div className="flex-1 bg-zinc-50 border border-zinc-100" />
                <div className="h-3 bg-zinc-100 w-1/2" />
              </div>
            )}

            {diagnosis && (
              <div className="flex flex-col gap-6 animate-fade-in-up">
                {/* Prediction card */}
                <div className="bg-white border border-zinc-200 p-6" id="result-prediction">
                  <p className="text-[11px] text-zinc-400 uppercase tracking-[0.2em] mb-3">
                    Classification
                  </p>
                  <div className="flex items-end justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-3 h-3 ${isMalignant ? "bg-black" : "bg-zinc-300"}`}
                      />
                      <span className="text-2xl font-semibold tracking-tight text-black">
                        {diagnosis.prediction}
                      </span>
                    </div>
                    <span className="text-sm font-mono text-zinc-500">
                      {confidencePct}%
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div className="mt-4">
                    <div className="h-1 bg-zinc-100 w-full">
                      <div
                        className="h-full bg-black transition-all duration-700 ease-out"
                        style={{ width: `${confidencePct}%` }}
                      />
                    </div>
                    <div className="flex justify-between mt-1.5">
                      <span className="text-[10px] text-zinc-400 font-mono">0.0</span>
                      <span className="text-[10px] text-zinc-400 font-mono">1.0</span>
                    </div>
                  </div>
                </div>

                {/* Biomarkers card */}
                <div className="bg-white border border-zinc-200 p-6" id="result-biomarkers">
                  <p className="text-[11px] text-zinc-400 uppercase tracking-[0.2em] mb-4">
                    Morphological Biomarkers
                  </p>

                  <div className="divide-y divide-zinc-100">
                    {/* N:C Ratio */}
                    <div className="flex items-center justify-between py-3">
                      <div>
                        <p className="text-xs font-medium text-black">
                          Nuclear-to-Cytoplasmic Ratio
                        </p>
                        <p className="text-[10px] text-zinc-400 mt-0.5">
                          Elevated ratios indicate proliferative activity
                        </p>
                      </div>
                      <span
                        className={`text-[11px] font-mono tracking-wider px-2.5 py-1 border ${
                          diagnosis.biological_indicators.nc_ratio === "High"
                            ? "border-black text-black bg-zinc-50"
                            : "border-zinc-200 text-zinc-400"
                        }`}
                      >
                        {diagnosis.biological_indicators.nc_ratio}
                      </span>
                    </div>

                    {/* Pleomorphism */}
                    <div className="flex items-center justify-between py-3">
                      <div>
                        <p className="text-xs font-medium text-black">
                          Pleomorphism
                        </p>
                        <p className="text-[10px] text-zinc-400 mt-0.5">
                          Irregular variation in cell size and shape
                        </p>
                      </div>
                      <span
                        className={`text-[11px] font-mono tracking-wider px-2.5 py-1 border ${
                          diagnosis.biological_indicators.pleomorphism === "Observed"
                            ? "border-black text-black bg-zinc-50"
                            : "border-zinc-200 text-zinc-400"
                        }`}
                      >
                        {diagnosis.biological_indicators.pleomorphism}
                      </span>
                    </div>

                    {/* Hyperchromasia */}
                    <div className="flex items-center justify-between py-3">
                      <div>
                        <p className="text-xs font-medium text-black">
                          Hyperchromasia
                        </p>
                        <p className="text-[10px] text-zinc-400 mt-0.5">
                          Dark-staining nuclei from dense chromatin packing
                        </p>
                      </div>
                      <span
                        className={`text-[11px] font-mono tracking-wider px-2.5 py-1 border ${
                          diagnosis.biological_indicators.hyperchromasia === "Detected"
                            ? "border-black text-black bg-zinc-50"
                            : "border-zinc-200 text-zinc-400"
                        }`}
                      >
                        {diagnosis.biological_indicators.hyperchromasia}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Case Analysis card */}
                <div className="bg-white border border-zinc-200 p-6" id="result-reasoning">
                  <p className="text-[11px] text-zinc-400 uppercase tracking-[0.2em] mb-4">
                    Case Analysis
                  </p>
                  <div className="flex flex-col gap-4">
                    {Object.entries(diagnosis.case_analysis)
                      .filter(([key]) => key !== "8_classification_percentage")
                      .sort()
                      .map(([key, value]) => {
                      // Clean up the key name for display (e.g., "0_pathogenesis" -> "Pathogenesis")
                      const label = key.split('_').slice(1).map(word => 
                        word.charAt(0).toUpperCase() + word.slice(1)
                      ).join(' ');
                      
                      return (
                        <div key={key} className="border-l-2 border-zinc-200 pl-4">
                          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-1">
                            {label}
                          </p>
                          <p className="text-sm text-zinc-700 leading-relaxed">
                            {value}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Layman explanation card */}
                {layman && (
                  <div className="bg-zinc-50 border border-zinc-200 p-6 animate-fade-in-up delay-200" id="result-layman">
                    <div className="flex items-center gap-2 mb-4">
                      <p className="text-[11px] text-zinc-400 uppercase tracking-[0.2em]">
                        Plain English
                      </p>
                      <span className="text-[9px] border border-zinc-300 text-zinc-400 px-1.5 py-0.5 uppercase tracking-wider">
                        For non-medical readers
                      </span>
                    </div>

                    {/* Verdict summary */}
                    <p className="text-sm text-black leading-relaxed font-medium mb-4">
                      {layman.headline}
                    </p>

                    <div className="border-t border-zinc-200 pt-4">
                      <p className="text-[11px] text-zinc-400 uppercase tracking-wider mb-3">
                        What each marker means
                      </p>
                      <ul className="flex flex-col gap-3">
                        {layman.bullets.map((bullet, i) => (
                          <li key={i} className="text-xs text-zinc-600 leading-relaxed">
                            {bullet}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <p className="text-[10px] text-zinc-400 mt-4 pt-4 border-t border-zinc-200 leading-relaxed">
                      ⚠ This tool is an educational AI prototype. It is not a substitute for a qualified pathologist or clinical diagnosis.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-200 bg-white mt-auto">
        <div className="mx-auto max-w-6xl flex items-center justify-between px-6 py-3">
          <p className="text-[10px] text-zinc-400 font-mono uppercase tracking-wider">
            OncoVision AI v1.0 — Biology for Engineering
          </p>
          <p className="text-[10px] text-zinc-400 font-mono">
            Model: gemini-2.5-flash · Multimodal LLM Inference
          </p>
        </div>
      </footer>
    </div>
  );
}
