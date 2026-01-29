import React from "react";
import UploadZone from "./components/UploadZone";
import GraphViewer from "./components/GraphViewer";
import { processPdf } from "./api";

type Status = "idle" | "loading" | "success" | "error";

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
      const { graphHtml: html } = await processPdf(file);
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
          <h1 className="title">PDF Character Graph</h1>
          <p className="subtitle">
            Upload a PDF book to visualize character relationships as an interactive graph.
          </p>
        </div>
      </div>

      <div className="row">
        <UploadZone disabled={status === "loading"} onPickFile={handlePickFile} />

        {status === "loading" && (
          <div className="card">
            <div style={{ fontWeight: 650, marginBottom: 6 }}>Processing...</div>
            <div className="hint">
              {fileName ? (
                <>
                  Analyzing <code>{fileName}</code> for characters and relationships.
                  This may take a few minutes for larger books.
                </>
              ) : (
                "Uploading and analyzing the PDF..."
              )}
            </div>
            <div className="loader" style={{ marginTop: 12 }} />
            <div style={{ marginTop: 12 }} className="btn-row">
              <button className="btn" type="button" onClick={reset}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {status === "error" && error && (
          <div className="card">
            <div className="error">{error}</div>
            <div style={{ marginTop: 12 }} className="btn-row">
              <button className="btn" type="button" onClick={reset}>
                Try again
              </button>
            </div>
          </div>
        )}

        {graphHtml && <GraphViewer graphHtml={graphHtml} />}
      </div>
    </div>
  );
}
