import React from "react";
import UploadZone from "./components/UploadZone";
import GraphViewer from "./components/GraphViewer";

type Status = "idle" | "loading" | "success" | "error";

async function postPdfForGraph(file: File): Promise<{ graphHtml: string }> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/process", {
    method: "POST",
    body: form
  });

  const contentType = res.headers.get("content-type") ?? "";
  const isJson = contentType.toLowerCase().includes("application/json");

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      if (isJson) {
        const body = (await res.json()) as { error?: string; detail?: string };
        detail = body.error || body.detail || detail;
      } else {
        const text = await res.text();
        detail = text?.slice(0, 500) || detail;
      }
    } catch {
      // ignore parsing errors; keep fallback message
    }
    throw new Error(detail);
  }

  if (!isJson) {
    throw new Error("Expected JSON response with { graphHtml }.");
  }

  const data = (await res.json()) as { graphHtml?: unknown };
  if (typeof data.graphHtml !== "string" || data.graphHtml.trim() === "") {
    throw new Error("Response JSON did not include a non-empty graphHtml string.");
  }
  return { graphHtml: data.graphHtml };
}

export default function App() {
  const [status, setStatus] = React.useState<Status>("idle");
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [graphHtml, setGraphHtml] = React.useState<string | null>(null);

  const handlePickFile = async (file: File) => {
    setStatus("loading");
    setError(null);
    setGraphHtml(null);
    setFileName(file.name);

    try {
      const { graphHtml: html } = await postPdfForGraph(file);
      setGraphHtml(html);
      setStatus("success");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "Something went wrong.");
    }
  };

  const reset = () => {
    setStatus("idle");
    setFileName(null);
    setError(null);
    setGraphHtml(null);
  };

  return (
    <div className="page">
      <div className="header">
        <div>
          <h1 className="title">PDF → Interactive Graph</h1>
          <p className="subtitle">
            Upload a PDF, we send it to <code>/api/process</code>, then render the
            returned <code>graphHtml</code>.
          </p>
        </div>
      </div>

      <div className="row">
        <UploadZone disabled={status === "loading"} onPickFile={handlePickFile} />

        {status === "loading" ? (
          <div className="card">
            <div style={{ fontWeight: 650, marginBottom: 6 }}>Processing…</div>
            <div className="hint">
              {fileName ? (
                <>
                  Uploading <code>{fileName}</code> and waiting for the graph.
                </>
              ) : (
                "Uploading and waiting for the graph."
              )}
            </div>
            <div style={{ marginTop: 12 }} className="btn-row">
              <button className="btn" type="button" onClick={reset}>
                Reset
              </button>
            </div>
          </div>
        ) : null}

        {status === "error" && error ? (
          <div className="card">
            <div className="error">{error}</div>
            <div style={{ marginTop: 12 }} className="btn-row">
              <button className="btn" type="button" onClick={reset}>
                Try again
              </button>
            </div>
          </div>
        ) : null}

        {graphHtml ? <GraphViewer graphHtml={graphHtml} /> : null}
      </div>
    </div>
  );
}

